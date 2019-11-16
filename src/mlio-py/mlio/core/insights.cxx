/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You
 * may not use this file except in compliance with the License. A copy of
 * the License is located at
 *
 *      http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific
 * language governing permissions and limitations under the License.
 */

#include "core/module.h"
#include "mlio/parser.h"
#include "mlio/util/number.h"
#include "mlio/util/string.h"
#include "tbb/tbb.h"

#include <pybind11/stl_bind.h>

#include <exception>
#include <functional>
#include <string>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace py = pybind11;

using namespace tbb;
using namespace pybind11::literals;

namespace mliopy {
namespace detail {
namespace {

class column_analysis {
    public:

    column_analysis(std::string name) 
        : rows_seen(0),
          numeric_mean(0.0), 
          numeric_count(0), 
          numeric_nan_count(0), 
          numeric_min(std::numeric_limits<double>::quiet_NaN()),
          numeric_max(std::numeric_limits<double>::quiet_NaN()),
          string_empty_count(0),
          string_only_whitespace_count(0),
          string_null_like_count(0),
          string_captured_unique_values_overflowed(false),
          null_empty_count(0),
          null_like_count(0), 
          null_whitespace_only_count(0),
          example_value("")
    { column_name = name; }

    std::string column_name;

    long rows_seen;
    double numeric_mean;
    long numeric_count;
    long numeric_nan_count;
    double numeric_min;
    double numeric_max;

    long string_empty_count;
    long string_only_whitespace_count;
    long string_null_like_count;
    std::unordered_set<std::string> string_captured_unique_values;
    bool string_captured_unique_values_overflowed;
    
    long null_empty_count;
    long null_like_count;
    long null_whitespace_only_count;

    std::string example_value;
};

class data_analysis {
    public:
    data_analysis(std::vector<detail::column_analysis> columns_) { columns = columns_; }
    std::vector<column_analysis> columns;
};

class column_analyzer {
    mlio::intrusive_ptr<mlio::example> * current_example;
    std::vector<detail::column_analysis> * columns;
    std::unordered_set<std::string> const *null_like_values;
    std::unordered_set<size_t> const *capture_columns;
    size_t max_capture_count;

    public:
    void operator()(const blocked_range<size_t>& r) const {
        for (size_t feature_idx = r.begin(); feature_idx != r.end(); ++feature_idx) {
            auto t = (*current_example)->features().at(feature_idx);
            column_analysis & feature_statistics = columns->at(feature_idx);
            
            auto dt = static_cast<mlio::dense_tensor*>(t.get());
            auto cells = dt->data().as<std::string>();

            auto should_capture = capture_columns->count(feature_idx) > 0;

            for (std::string as_string : cells) {
                feature_statistics.rows_seen += 1;
                feature_statistics.example_value = as_string;
                feature_statistics.numeric_min += 1;
                feature_statistics.string_null_like_count += 1;

                if (should_capture) {
                    if (feature_statistics.string_captured_unique_values.size() < max_capture_count) {
                        feature_statistics.string_captured_unique_values.insert(as_string);
                    } else if (feature_statistics.string_captured_unique_values.count(as_string) == 0) {
                        // If the value isn't present but we're not adding it because we're at a limit
                        // then we should flag that we have overflowed.
                        feature_statistics.string_captured_unique_values_overflowed = true;
                    }
                }

                // Numeric analyzers
                double as_float;
                if (mlio::try_parse_float(mlio::v1::float_parse_params{as_string}, as_float) != mlio::parse_result::ok) {
                    feature_statistics.numeric_nan_count += 1;
                } else {
                    feature_statistics.numeric_count += 1;
                    if (isnan(feature_statistics.numeric_min) || as_float < feature_statistics.numeric_min) {
                        feature_statistics.numeric_min = as_float;
                    }
                    if (isnan(feature_statistics.numeric_max) || as_float > feature_statistics.numeric_max) {
                        feature_statistics.numeric_max = as_float;
                    }
                }

                // String analyzers.
                if (as_string.size() == 0) {
                    feature_statistics.string_empty_count += 1;
                }

                if (mlio::only_whitespace(as_string)) {
                    feature_statistics.string_only_whitespace_count += 1;
                }

                auto lowercase_string = as_string;
                std::transform(as_string.begin(), as_string.end(), lowercase_string.begin(), ::tolower);
                if (mlio::matches(lowercase_string, null_like_values)) {
                    feature_statistics.string_null_like_count += 1;
                }
            }
        }
    }

    column_analyzer(mlio::intrusive_ptr<mlio::example> * current_example_,
                   std::vector<detail::column_analysis> * columns_,
                   std::unordered_set<std::string> const *null_like_values_,
                   std::unordered_set<size_t> const *capture_columns_,
                   size_t max_capture_count_) 
    : current_example(current_example_), 
      columns(columns_), 
      null_like_values(null_like_values_),
      capture_columns(capture_columns_),
      max_capture_count(max_capture_count_) {}
};

pybind11::object
analyze_dataset(
    mlio::intrusive_ptr<mlio::data_reader> reader_,
    std::unordered_set<std::string> const *null_like_values,
    std::unordered_set<size_t> const *capture_columns,
    size_t max_capture_count = 5000)
{   
    // Iterate over the entire dataset.
    bool end_of_dataset = false;

    // Stores the data that our parallel function will operate on.
    mlio::intrusive_ptr<mlio::example> exm = nullptr;
    std::vector<column_analysis> columns;

    // Analyzer that is used in the parallel function.
    auto analyzer = column_analyzer(&exm, 
                                    &columns, 
                                    null_like_values,
                                    capture_columns,
                                    max_capture_count);

    while (! end_of_dataset) {
        exm = reader_->read_example();
        if (exm == nullptr) {
            end_of_dataset = true;
            continue;
        }
        
        // On first feature - allocate the column analysis data structures.
        if (columns.empty()) {
            columns.reserve(exm->features().size());
            for (mlio::feature_desc desc : exm->get_schema().descriptors()) {
                auto statistics = column_analysis(desc.name());
                columns.push_back(statistics);
            }

            for (mlio::intrusive_ptr<mlio::tensor> t : exm->features()) {
                switch (t->dtype()) {
                    case mlio::data_type::string:
                        break;
                    case mlio::data_type::size:
                    case mlio::data_type::float16:
                    case mlio::data_type::float32:
                    case mlio::data_type::float64:
                    case mlio::data_type::sint8:
                    case mlio::data_type::sint16:
                    case mlio::data_type::sint32:
                    case mlio::data_type::sint64:
                    case mlio::data_type::uint8:
                    case mlio::data_type::uint16:
                    case mlio::data_type::uint32:
                    case mlio::data_type::uint64:
                        throw std::runtime_error("Data insights only works with string tensors.");
                }
            }
        }

        parallel_for(blocked_range<size_t>(0, columns.size()), analyzer);
    }
    return pybind11::cast(detail::data_analysis(columns));
}

}  // namespace
}  // namespace detail

void
register_insights(py::module &m)
{

    std::vector<std::pair<const char *, double detail::column_analysis::*>> double_statistic_names = {
        std::make_pair("numeric_mean", &detail::column_analysis::numeric_mean),
        std::make_pair("numeric_min", &detail::column_analysis::numeric_min),
        std::make_pair("numeric_max", &detail::column_analysis::numeric_max),
    };

    std::vector<std::pair<const char *, long detail::column_analysis::*>> long_statistic_names = {
        std::make_pair("rows_seen", &detail::column_analysis::rows_seen),
        std::make_pair("numeric_count", &detail::column_analysis::numeric_count),
        std::make_pair("numeric_nan_count", &detail::column_analysis::numeric_nan_count),
        std::make_pair("string_empty_count", &detail::column_analysis::string_empty_count),
        std::make_pair("string_only_whitespace_count", &detail::column_analysis::string_only_whitespace_count),
        std::make_pair("string_null_like_count", &detail::column_analysis::string_null_like_count),
        std::make_pair("null_empty_count", &detail::column_analysis::null_empty_count),
        std::make_pair("null_like_count", &detail::column_analysis::null_like_count),
        std::make_pair("null_whitespace_only_count", &detail::column_analysis::null_whitespace_only_count),
    };

    std::vector<std::pair<const char *, std::string detail::column_analysis::*>> string_statistic_names = {
        std::make_pair("example_value", &detail::column_analysis::example_value),
    };

    auto ca_class = py::class_<mliopy::detail::column_analysis>(m, "ColumnAnalysis");
    ca_class.def_readwrite("column_name", &detail::column_analysis::column_name);
    ca_class.def_readwrite("string_captured_unique_values", &detail::column_analysis::string_captured_unique_values);
    ca_class.def_readwrite("string_captured_unique_values_overflowed", &detail::column_analysis::string_captured_unique_values_overflowed);

    
    for (auto name_method_pair : long_statistic_names) {
        auto name = name_method_pair.first;
        auto method = name_method_pair.second;
        ca_class.def_readwrite(name, method);
    }

    for (auto name_method_pair : double_statistic_names) {
        auto name = name_method_pair.first;
        auto method = name_method_pair.second;
        ca_class.def_readwrite(name, method);
    }

    for (auto name_method_pair : string_statistic_names) {
        auto name = name_method_pair.first;
        auto method = name_method_pair.second;
        ca_class.def_readwrite(name, method);
    }

    ca_class.def("to_dict", 
        [=](const detail::column_analysis &ca) {
            py::dict result;

            for (auto name_method_pair : long_statistic_names) {
                auto name = name_method_pair.first;
                auto method = name_method_pair.second;
                long value = ca.*method;
                result[name] = std::to_string(value);
            }

            for (auto name_method_pair : double_statistic_names) {
                auto name = name_method_pair.first;
                auto method = name_method_pair.second;
                double value = (ca.*method);
                result[name] = std::to_string(value);
            }

            for (auto name_method_pair : string_statistic_names) {
                auto name = name_method_pair.first;
                auto method = name_method_pair.second;
                std::string value = (ca.*method);
                result[name] = value;
            }

            return result;
        });

    ca_class.def("__repr__", 
        [](const mliopy::detail::column_analysis &ca) {
            return "ColumnAnalysis(" + ca.column_name + ")";
        });

    py::class_<mliopy::detail::data_analysis>(
        m, "DataAnalysis").def_readwrite("columns", &detail::data_analysis::columns);

    m.def("analyze_dataset", &detail::analyze_dataset, "Analyzes a dataset");

}

}  // namespace mliopy
