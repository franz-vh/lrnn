#ifndef __MMM_MMMMPVISUALIZERCONFIGURATION_H_
#define __MMM_MMMMPVISUALIZERCONFIGURATION_H_

#include <filesystem>
#include <string>

#include <MMM/MMMCore.h>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include <VirtualRobot/RuntimeEnvironment.h>

#include <MMM/Motion/Segmentation/AbstractMotionSegment.h>
#include <MMM/Motion/Segmentation/MotionSegment.h>
#include <MMM/Motion/Segmentation/MotionRecordingSegment.h>
#include <MMM/Motion/Segmentation/Segmentation.h>
#include <MMM/Motion/Annotation/ActionLabel/ActionLabelAnnotation.h>

#include <SimoxUtility/algorithm/string.h>

#include "../common/ApplicationBaseConfiguration.h"

class MMMSegmentwiseCSVExporterConfiguration : public ApplicationBaseConfiguration
{

public:
    std::vector<std::filesystem::path> motionFiles;
    std::filesystem::path outputDir;
    std::filesystem::path objectPath;
    bool overwrite;

    MMMSegmentwiseCSVExporterConfiguration() : ApplicationBaseConfiguration()
    {
    }

    bool processCommandLine(int argc, char *argv[])
    {
        VirtualRobot::RuntimeEnvironment::considerKey("motionFile", "motion file");
        VirtualRobot::RuntimeEnvironment::considerKey("outputDir", "output directory");
        VirtualRobot::RuntimeEnvironment::considerKey("objectPath", "object directory");
        VirtualRobot::RuntimeEnvironment::considerFlag("overwrite", "");

        VirtualRobot::RuntimeEnvironment::processCommandLine(argc,argv);
        VirtualRobot::RuntimeEnvironment::print();

        std::string motionFiles = getParameter("motionFile", true, false);
        if (motionFiles.find(";") != std::string::npos) {
            for (const std::string &motionFile : simox::alg::split(motionFiles, ";"))
                this->motionFiles.push_back(motionFile);
        }
        else this->motionFiles.push_back(motionFiles);
        objectPath = getParameter("objectPath", false, false);
        outputDir = getParameter("outputDir", false, false);
        if (outputDir.empty())
            outputDir = std::filesystem::path(this->motionFiles.at(0)).parent_path();

        overwrite = VirtualRobot::RuntimeEnvironment::hasFlag("overwrite");

        return valid;
    }

    void print()
    {
        MMM_INFO << "*** MMMMPVisualizer Configuration ***" << std::endl;
        std::cout << "Output directory: " << outputDir << std::endl;
    }
};


#endif
