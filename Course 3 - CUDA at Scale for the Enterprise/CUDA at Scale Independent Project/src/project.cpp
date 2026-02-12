/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

bool printfNPPinfo(int argc, char *argv[])
{
       const NppLibraryVersion *libVer = nppGetLibVersion();

       printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
              libVer->build);

       int driverVersion, runtimeVersion;
       cudaDriverGetVersion(&driverVersion);
       cudaRuntimeGetVersion(&runtimeVersion);

       printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
              (driverVersion % 100) / 10);
       printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
              (runtimeVersion % 100) / 10);

       // Min spec is SM 1.0 devices
       bool bVal = checkCudaCapabilities(1, 0);
       return bVal;
}

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// Windows implementation using std::filesystem
void findAllPGMFiles(const std::string &directory, std::vector<std::string> &files)
{
       if (!fs::exists(directory))
       {
              std::cerr << "Directory does not exist: " << directory << std::endl;
              return;
       }

       try
       {
              for (const auto &entry : fs::recursive_directory_iterator(directory))
              {
                     if (entry.is_regular_file())
                     {
                            std::string filename = entry.path().string();
                            if (filename.length() >= 4 && filename.substr(filename.length() - 4) == ".pgm")
                            {
                                   files.push_back(filename);
                            }
                     }
              }
       }
       catch (const fs::filesystem_error &e)
       {
              std::cerr << "Filesystem error: " << e.what() << std::endl;
       }
}
#else
// Linux implementation using dirent
void findAllPGMFilesRecursive(const std::string &directory, std::vector<std::string> &files)
{
       DIR *dir = opendir(directory.c_str());
       if (!dir)
              return;

       struct dirent *entry;
       while ((entry = readdir(dir)) != nullptr)
       {
              std::string name = entry->d_name;
              if (name == "." || name == "..")
                     continue;

              std::string fullPath = directory + "/" + name;

              struct stat statbuf;
              if (stat(fullPath.c_str(), &statbuf) == 0)
              {
                     if (S_ISDIR(statbuf.st_mode))
                     {
                            findAllPGMFilesRecursive(fullPath, files);
                     }
                     else if (S_ISREG(statbuf.st_mode))
                     {
                            if (name.length() >= 4 && name.substr(name.length() - 4) == ".pgm")
                            {
                                   files.push_back(fullPath);
                            }
                     }
              }
       }
       closedir(dir);
}

void findAllPGMFiles(const std::string &directory, std::vector<std::string> &files)
{
       findAllPGMFilesRecursive(directory, files);
}
#endif

std::string getOutputFilename(const std::string &inputFile, int augNumber)
{
       std::string result = inputFile;
       std::string::size_type dot = result.rfind('.');

       if (dot != std::string::npos)
       {
              result = result.substr(0, dot);
       }

       result += "_aug" + std::to_string(augNumber) + ".pgm";
       return result;
}

void processImage(const std::string &inputFile, const std::string &outputDir,
                  std::mt19937 &rng, int imageIndex)
{
       std::cout << "\nProcessing: " << inputFile << std::endl;

       const int augmentationsPerImage = 2;

       // Load the image
       npp::ImageCPU_8u_C1 oHostSrc;
       npp::loadImage(inputFile, oHostSrc);
       npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

       NppiSize oSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};

       for (int aug = 0; aug < augmentationsPerImage; aug++)
       {
              npp::ImageNPP_8u_C1 oDeviceResult(oDeviceSrc.width(), oDeviceSrc.height());

              if (aug == 0)
              {
                     // Horizontal flip
                     NPP_CHECK_NPP(nppiMirror_8u_C1R(
                         oDeviceSrc.data(), oDeviceSrc.pitch(),
                         oDeviceResult.data(), oDeviceResult.pitch(),
                         oSize, NPP_HORIZONTAL_AXIS));
              }
              else
              {
                     // Vertical flip
                     NPP_CHECK_NPP(nppiMirror_8u_C1R(
                         oDeviceSrc.data(), oDeviceSrc.pitch(),
                         oDeviceResult.data(), oDeviceResult.pitch(),
                         oSize, NPP_VERTICAL_AXIS));
              }

              // Copy result to host and save
              npp::ImageCPU_8u_C1 oHostDst(oDeviceResult.size());
              oDeviceResult.copyTo(oHostDst.data(), oHostDst.pitch());

              std::string outputFile = getOutputFilename(inputFile, aug);

              saveImage(outputFile, oHostDst);
              std::cout << "  Saved: " << outputFile << std::endl;

              nppiFree(oDeviceResult.data());
       }

       nppiFree(oDeviceSrc.data());
}

int main(int argc, char *argv[])
{
       printf("%s Starting...\n\n", argv[0]);

       try
       {
              findCudaDevice(argc, (const char **)argv);

              if (printfNPPinfo(argc, argv) == false)
              {
                     exit(EXIT_SUCCESS);
              }

              // Determine input directory
              std::string inputDir;
              if (checkCmdLineFlag(argc, (const char **)argv, "input"))
              {
                     char *dirPath;
                     getCmdLineArgumentString(argc, (const char **)argv, "input", &dirPath);
                     inputDir = dirPath;
              }
              else
              {
                     inputDir = "../data/faces";
              }

              // Find all PGM files
              std::vector<std::string> pgmFiles;
              std::cout << "Searching for PGM files in: " << inputDir << std::endl;
              findAllPGMFiles(inputDir, pgmFiles);

              if (pgmFiles.empty())
              {
                     std::cerr << "No PGM files found in directory: " << inputDir << std::endl;
                     std::cerr << "Make sure the directory exists and contains .pgm files (or subdirectories with .pgm files)." << std::endl;
                     exit(EXIT_FAILURE);
              }

              std::cout << "\nFound " << pgmFiles.size() << " PGM files to process." << std::endl;
              std::cout << "Generating synthetic training data with horizontal and vertical flips...\n"
                        << std::endl;

              // Initialize random number generator
              unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
              std::mt19937 rng(seed);

              // Process each image
              int processedCount = 0;
              for (size_t i = 0; i < pgmFiles.size(); i++)
              {
                     try
                     {
                            processImage(pgmFiles[i], inputDir, rng, i);
                            processedCount++;
                     }
                     catch (npp::Exception &rException)
                     {
                            std::cerr << "Error processing " << pgmFiles[i] << ": "
                                      << rException << std::endl;
                            continue;
                     }
                     catch (...)
                     {
                            std::cerr << "Unknown error processing " << pgmFiles[i] << std::endl;
                            continue;
                     }
              }

              std::cout << "\n========================================" << std::endl;
              std::cout << "Processing complete!" << std::endl;
              std::cout << "Processed " << processedCount << " images." << std::endl;
              std::cout << "Generated " << (processedCount * 2) << " augmented images." << std::endl;
              std::cout << "========================================\n"
                        << std::endl;

              exit(EXIT_SUCCESS);
       }
       catch (npp::Exception &rException)
       {
              std::cerr << "Program error! The following exception occurred: \n";
              std::cerr << rException << std::endl;
              std::cerr << "Aborting." << std::endl;

              exit(EXIT_FAILURE);
       }
       catch (...)
       {
              std::cerr << "Program error! An unknown type of exception occurred. \n";
              std::cerr << "Aborting." << std::endl;

              exit(EXIT_FAILURE);
              return -1;
       }

       return 0;
}
