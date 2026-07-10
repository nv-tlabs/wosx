/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <string>
#include <vector>
#include <filesystem>

namespace wosx {

class GPULibraryPaths {
public:
    // constructor
    GPULibraryPaths(const std::string& wosxDirectoryPath_,
                    const std::string& wosxPdeDirectoryPath_="");

    // getters
    std::string getFcpwGpuDirectoryPath() const;
    std::string getWosxGpuDirectoryPath() const;
    std::vector<std::string> getSearchPaths() const;
    std::string getFcpwModule() const;
    std::string getWosxModule() const;
    std::string getWosxPdeModule() const;
    std::vector<std::string> getModules() const;

private:
    // members
    std::vector<std::string> searchPaths;
    std::filesystem::path wosxDirectoryPath;
    std::filesystem::path fcpwGpuDirectoryPath;
    std::filesystem::path wosxGpuDirectoryPath;
    std::string wosxPdeModule;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

inline GPULibraryPaths::GPULibraryPaths(const std::string& wosxDirectoryPath_,
                                        const std::string& wosxPdeDirectoryPath_)
{
    fcpwGpuDirectoryPath = wosxDirectoryPath_ + "/deps/fcpw/include/fcpw/gpu";
    wosxGpuDirectoryPath = wosxDirectoryPath_ + "/include/wosx/gpu";
    searchPaths.emplace_back(fcpwGpuDirectoryPath.string());
    searchPaths.emplace_back(wosxGpuDirectoryPath.string());
    if (!wosxPdeDirectoryPath_.empty()) {
        wosxPdeModule = wosxPdeDirectoryPath_ + "/wosx-pde-definition.slang";
        searchPaths.emplace_back(wosxPdeDirectoryPath_);

    } else {
        wosxPdeModule = "";
    }
}

inline std::string GPULibraryPaths::getFcpwGpuDirectoryPath() const
{
    return fcpwGpuDirectoryPath.string();
}

inline std::string GPULibraryPaths::getWosxGpuDirectoryPath() const
{
    return wosxGpuDirectoryPath.string();
}

inline std::vector<std::string> GPULibraryPaths::getSearchPaths() const
{
    return searchPaths;
}

inline std::string GPULibraryPaths::getFcpwModule() const
{
    return (fcpwGpuDirectoryPath / "fcpw.slang").string();
}

inline std::string GPULibraryPaths::getWosxModule() const
{
    return (wosxGpuDirectoryPath / "wosx.slang").string();
}

inline std::string GPULibraryPaths::getWosxPdeModule() const
{
    return wosxPdeModule;
}

inline std::vector<std::string> GPULibraryPaths::getModules() const
{
    std::vector<std::string> modules;
    modules.emplace_back(getFcpwModule());
    modules.emplace_back(getWosxModule());
    if (!wosxPdeModule.empty()) {
        modules.emplace_back(wosxPdeModule);
    }

    return modules;
}

} // wosx
