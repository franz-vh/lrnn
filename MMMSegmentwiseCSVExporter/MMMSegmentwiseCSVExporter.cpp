#include "MMMSegmentwiseCSVExporterConfiguration.h"
#include <MMM/Motion/MotionReaderXML.h>
#include <MMM/Motion/Motion.h>
#include <string>
#include <vector>
#include <tuple>
#include <SimoxUtility/algorithm/string.h>
#include <SimoxUtility/math/convert.h>
#include <SimoxUtility/filesystem/make_relative.h>
#include <VirtualRobot/Robot.h>
#include <MMM/Model/LoadModelStrategy.h>
#include <MMM/Model/ModelProcessor.h>

#include "../../common/HandleMotionsWithoutModel.h"
#include <VirtualRobot/XML/RobotIO.h>

using namespace MMM;



class WriteFileWrapper
{
public:
    bool addHeaderCheck(const std::initializer_list<std::string>& names, bool identifier = false)
    {
        for (const std::string& name : names)
        {
            if (!addHeaderCheck(name, identifier))
                return false;
        }
        return true;
    }

    bool addHeaderCheck(const std::string& name, bool identifier = false)
    {
        if (map.find(name) != map.end())
            return false;
        else
        {
            addHeader(name, identifier);
            return true;
        }
    }

    template<typename T, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
    void add(const std::string& name, T value, bool throwError = true)
    {
        add(currentIndex(), name, value, throwError);
    }

    template<typename T, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
    void add(size_t index, const std::string& name, T value, bool throwError = true)
    {
        std::string valueStr = simox::alg::to_string(value);
        add(index, name, valueStr, throwError);
    }

    void add(const std::string& name, const Eigen::Matrix4f &pose, bool throwError = true)
    {
        add(currentIndex(), name, pose, throwError);
    }

    void add(size_t index, const std::string& name, const Eigen::Matrix4f &pose, bool throwError = true)
    {
        Eigen::Vector3f position = simox::math::mat4f_to_pos(pose);
        Eigen::Quaternionf orientation = simox::math::mat4f_to_quat(pose).normalized();
	add(index, name, position, orientation, throwError);
    }

    // function by Franz
    void add(const std::string& name, const Eigen::Vector3f &position, const Eigen::Quaternionf &orientation, bool throwError = true)
    {
	add(currentIndex(), name, position, orientation, throwError);
    }

    // function by Franz
    void add(size_t index, const std::string& name, const Eigen::Vector3f &position, const Eigen::Quaternionf &orientation, bool throwError = true)
    {
	add(index, name + "[x]", position(0), throwError);
        add(index, name + "[y]", position(1), throwError);
        add(index, name + "[z]", position(2), throwError);
        add(index, name + "[qw]", orientation.w(), throwError);
        add(index, name + "[qx]", orientation.x(), throwError);
        add(index, name + "[qy]", orientation.y(), throwError);
        add(index, name + "[qz]", orientation.z(), throwError);
    }

    void add(const std::string& name, const std::string& value, bool throwError = true)
    {
        add(currentIndex(), name, value, throwError);
    }

    void add(size_t index, const std::string& name, const std::string& value, bool throwError = true)
    {
        if (value.empty()) return;

        if (map.find(name) == map.end())
        {
            if (throwError) throw std::runtime_error("Unknown name " + name + " with value " + value);
            else addHeader(name);
        }

        map.at(name).value.insert(std::make_pair(index, value));

        maxIndex = std::max(index, maxIndex);
    }

    void writeCSV(const std::filesystem::path& filePath, bool emptyLine = false) const
    {
        std::ofstream file(filePath);

        if (!file) {
            MMM_ERROR << filePath << " could not be opened" << std::endl;
            return;
        }

        // Append header
        for (size_t col = 0; col < order.size(); col++)
        {
            const KeyValues &kv = map.at(order.at(col).get());
            file << kv.name;
            if (col < order.size() - 1) file << ",";
        }
        if (emptyLine) file << "\n";

        // Append values
        for (size_t row = 0; row < maxIndex; row++)
        {
            for (size_t col = 0; col < order.size(); col++)
            {
                const KeyValues &kv = map.at(order.at(col).get());
                file << kv.at(row);
                if (col < order.size() - 1) file << ",";
            }
            file << "\n";
        }

        file.close();
    }

    size_t currentIndex() const
    {
        return maxIndex;
    }

    size_t nextIndex() const
    {
        return maxIndex + 1;
    }

    void increment()
    {
        maxIndex++;
    }

    std::string getIdentifier() const
    {
        return getIdentifier(currentIndex());
    }

    std::string getIdentifier(size_t index) const
    {
        std::stringstream ss;
        bool set = false;
        for (size_t col = 0; col < order.size(); col++)
        {
            const std::string &key = order.at(col).get();
            const KeyValues &kv = map.at(key);
            if (kv.identifier)
            {
                if (set) ss << "-_-";
                else set = true;
                ss << kv.at(index, "0");
            }
        }
        return ss.str();
    }

private:
    struct KeyValues
    {
        std::string name;
        std::map<size_t, std::string> value;
        bool identifier = false;

        const std::string at(size_t index, const std::string& defaultValue = std::string()) const
        {
            if (value.find(index) == value.end())
                return defaultValue;
            else
                return value.at(index);
        }
    };

    void addHeader(const std::string& name, bool identifier = false) {
        KeyValues kv;
        kv.name = name;
        kv.identifier = identifier;
        map.insert(std::make_pair(kv.name, std::move(kv)));
        order.push_back(std::ref(map.at(name).name));
    }

    std::vector<std::reference_wrapper<std::string>> order;
    std::map<std::string, KeyValues> map;
    size_t maxIndex = 0;

};

// function by Franz
void save(MotionRecordingPtr motions, MotionPtr humanMotion, VirtualRobot::RobotPtr mmmModel,
		   WriteFileWrapper &w, const std::string &filename, const std::filesystem::path &outputDir) {
	w.addHeaderCheck({ "timestep" }, true);
	MotionRecordingSegmentPtr segment = motions->getSegment();
	MotionSegmentationPtr leftSegmentation;
        MotionSegmentationPtr rightSegmentation;
	std::map<float, MotionSegmentPtr> leftSegments;
	std::map<float, MotionSegmentPtr>::iterator leftIt;
	std::map<float, MotionSegmentPtr> rightSegments;
	std::map<float, MotionSegmentPtr>::iterator rightIt;
	MotionSegmentationList leftSubSegmentations;
	std::map<float, MotionSegmentPtr> leftSubSegments;
	std::map<float, MotionSegmentPtr>::iterator leftSubIt;
	MotionSegmentationList rightSubSegmentations;
        std::map<float, MotionSegmentPtr> rightSubSegments;
        std::map<float, MotionSegmentPtr>::iterator rightSubIt;
	if (segment) {
	    w.addHeaderCheck({ "left_action",  "left_subaction",  "left_main_object",  "left_main_object_2",
                               "left_source_object",  "left_target_object",  "left_failure",
                               "right_action", "right_subaction", "right_main_object", "right_main_object_2",
                               "right_source_object", "right_target_object", "right_failure" }, true);
	    for (const auto &segmentation : segment->getSegmentations()) {
	    	std::string desc = segmentation->getDescription();
            	if (desc == "Left" || desc == "Left hand actions") {
                    leftSegmentation = segmentation;
                }
                else if (desc == "Right" || desc == "Right hand actions") {
                    rightSegmentation = segmentation;
                }
                else {
                    MMM_ERROR << "Segmentation " << desc << " is unknown!" << std::endl;
                    continue;
                }
	    }
	    leftSegments = leftSegmentation->getSegments();
	    leftIt = leftSegments.begin();
	    rightSegments = rightSegmentation->getSegments();
	    rightIt = rightSegments.begin();
	    leftSubSegmentations = leftIt->second->getSegmentations();
	    leftSubIt = leftIt;
	    if (leftSubSegmentations.size() > 0) {
	        leftSubSegments = leftSubSegmentations[0]->getSegments();
	        leftSubIt = leftSubSegments.begin();
	    }
	    rightSubSegmentations = rightIt->second->getSegmentations();
            rightSubIt = rightIt;
            if (rightSubSegmentations.size() > 0) {
                rightSubSegments = rightSubSegmentations[0]->getSegments();
                rightSubIt = rightSubSegments.begin();
            }
	}
	for (const std::string &bodyPartName : { "left_hand", "right_hand", "head", "torso" }) {
            w.addHeaderCheck({ bodyPartName + "[x]",  bodyPartName + "[y]",  bodyPartName + "[z]",
                               bodyPartName + "[qw]", bodyPartName + "[qx]", bodyPartName + "[qy]", bodyPartName + "[qz]"}, true);
        }
	std::vector<float> timesteps = humanMotion->getTimesteps();
	for (unsigned int i = 0; i < timesteps.size(); i++) {
	    float timestep = timesteps[i];
	    w.add("timestep", timestep);
	    for (const auto &motion : *motions) {
		if (motion->getName() == humanMotion->getName()) {
		    motion->initializeModel(mmmModel, timestep);
		    Eigen::Matrix4f leftTorsoPose = mmmModel->getRobotNode("LSCsegment_joint")->getGlobalPose();
		    Eigen::Matrix4f rightTorsoPose = mmmModel->getRobotNode("RSCsegment_joint")->getGlobalPose();
		    Eigen::Vector3f torsoPosition = (simox::math::mat4f_to_pos(leftTorsoPose) + simox::math::mat4f_to_pos(rightTorsoPose)) / 2;
        	    Eigen::Quaternionf torsoOrientation = simox::math::mat4f_to_quat(leftTorsoPose).normalized();
		    w.add("torso", torsoPosition, torsoOrientation);
		    Eigen::Matrix4f leftHeadPose = mmmModel->getRobotNode("LeftEyeSegmentX_joint")->getGlobalPose();
                    Eigen::Matrix4f rightHeadPose = mmmModel->getRobotNode("RightEyeSegmentX_joint")->getGlobalPose();
                    Eigen::Vector3f headPosition = (simox::math::mat4f_to_pos(leftHeadPose) + simox::math::mat4f_to_pos(rightHeadPose)) / 2;
                    Eigen::Quaternionf headOrientation = simox::math::mat4f_to_quat(leftHeadPose).normalized();
                    w.add("head", headPosition, headOrientation);
                    Eigen::Matrix4f leftHandPose = mmmModel->getRobotNode("Hand L TCP")->getGlobalPose();
                    w.add("left_hand", leftHandPose);
                    Eigen::Matrix4f rightHandPose = mmmModel->getRobotNode("Hand R TCP")->getGlobalPose();
                    w.add("right_hand", rightHandPose);
		}
		else if (motion->getName() != "kitchen_sideboard" && motion->getName() != "kitchen_sideboard_long"
			 && motion->getName() != "go_pro" && motion->getName() != "go_pro_new"
			 && motion->getName() != "kinect_azure_right" && motion->getName() != "kinect_azure_new_right"
			 && motion->getName() != "kinect_azure_new_left" && motion->getName() != "kinect_azure_new_front"
			 && motion->getName() != "kinect_azure_front" && motion->getName() != "kinect_azure_left") {
		    Eigen::Matrix4f pose = motion->getRootPose(timestep);
                    w.add(motion->getName(), pose, false);
		}
	    }
	    if (segment) {
	        while (timestep >= leftIt->second->getEndTimestep()) {
		    leftIt++;
		    if (leftIt == leftSegments.end()) {
		        return;
		    }
		    leftSubSegmentations = leftIt->second->getSegmentations();
		    leftSubIt = leftIt;
        	    if (leftSubSegmentations.size() > 0) {
            	        leftSubSegments = leftSubSegmentations[0]->getSegments();
            	        leftSubIt = leftSubSegments.begin();
        	    }
	        }
	        while (timestep >= leftSubIt->second->getEndTimestep())
                    leftSubIt++;
	        while (timestep >= rightIt->second->getEndTimestep()) {
		    rightIt++;
		    rightSubSegmentations = rightIt->second->getSegmentations();
                    rightSubIt = rightIt;
                    if (rightSubSegmentations.size() > 0) {
                        rightSubSegments = rightSubSegmentations[0]->getSegments();
                        rightSubIt = rightSubSegments.begin();
		    }
	        }
	        while (timestep >= rightSubIt->second->getEndTimestep())
                    rightSubIt++;
                auto leftActionAnnotation = leftIt->second->getDerivedAnnotation<ActionLabelAnnotation>();
                if (leftActionAnnotation) {
		    w.add("left_action", leftActionAnnotation->getLabelStr());
	            w.add("left_main_object", leftActionAnnotation->getMainObjectStr());
		    w.add("left_main_object_2", leftActionAnnotation->getSecondMainObjectStr());
		    w.add("left_source_object", leftActionAnnotation->getSourceStr());
		    w.add("left_target_object", leftActionAnnotation->getTargetStr());
		    w.add("left_failure", leftActionAnnotation->getFailureStr());
	        }
	        else {
                    //MMM_ERROR << "No action annotation at " << leftIt->first << " for left hand" << std::endl;
		    w.add("left_action", "Unknown");
                    w.add("left_main_object", "Unknown");
                    w.add("left_main_object_2", "Unknown");
                    w.add("left_source_object", "Unknown");
                    w.add("left_target_object", "Unknown");
                    w.add("left_failure", "Unknown");
                }
	        auto leftSubActionAnnotation = leftSubIt->second->getDerivedAnnotation<ActionLabelAnnotation>();
                if (leftSubActionAnnotation)
                    w.add("left_subaction", leftSubActionAnnotation->getLabelStr());
                else {
                    //MMM_ERROR << "No subaction annotation at " << leftSubIt->first << " for left hand" << std::endl;
                    w.add("left_subaction", "Unknown");
                }
	        auto rightActionAnnotation = rightIt->second->getDerivedAnnotation<ActionLabelAnnotation>();
	        if (rightActionAnnotation) {
                    w.add("right_action", rightActionAnnotation->getLabelStr());
                    w.add("right_main_object", rightActionAnnotation->getMainObjectStr());
                    w.add("right_main_object_2", rightActionAnnotation->getSecondMainObjectStr());
                    w.add("right_source_object", rightActionAnnotation->getSourceStr());
                    w.add("right_target_object", rightActionAnnotation->getTargetStr());
                    w.add("right_failure", rightActionAnnotation->getFailureStr());
                }
                else {
                    //MMM_ERROR << "No action annotation at " << rightIt->first << " for right hand" << std::endl;
                    w.add("right_action", "Unknown");
                    w.add("right_main_object", "Unknown");
                    w.add("right_main_object_2", "Unknown");
                    w.add("right_source_object", "Unknown");
                    w.add("right_target_object", "Unknown");
                    w.add("right_failure", "Unknown");
                }
	        auto rightSubActionAnnotation = rightSubIt->second->getDerivedAnnotation<ActionLabelAnnotation>();
                if (rightSubActionAnnotation)
                    w.add("right_subaction", rightSubActionAnnotation->getLabelStr());
                else {
                    //MMM_ERROR << "No subaction annotation at " << rightSubIt->first << " for right hand" << std::endl;
                    w.add("right_subaction", "Unknown");
                }
	    }
	    w.increment();
	}
}

void saveRecursive(AbstractMotionSegmentPtr segment, MotionRecordingPtr motions, MotionPtr humanMotion, VirtualRobot::RobotPtr mmmModel,
                   WriteFileWrapper &w, const std::string &filename, const std::filesystem::path &outputDir,
                   unsigned int maxLevel, unsigned int level = 0, std::shared_ptr<ActionLabelAnnotation> parentAction = nullptr) {
    if (level == maxLevel) return;
    for (const auto &segmentation : segment->getSegmentations()) {
        for (const auto &subsegment : segmentation->getSegments()) {
            std::string desc = segmentation->getDescription();
            std::string nodeSetName;
            if (desc.empty())
                if (auto segmentation = segment->getParentSegmentation())
                    desc = segmentation->getDescription();
            if (desc == "Left" || desc == "Left hand actions") {
                nodeSetName = "LeftArm-7dof";
            }
            else if (desc == "Right" || desc == "Right hand actions") {
                nodeSetName = "RightArm-7dof";
            }
            else if (auto model = humanMotion->getModel()) {
                if (!model->hasRobotNodeSet(desc)) {
                    MMM_ERROR << "Model has no robot node set " << desc << std::endl;
                    continue;
                }
                nodeSetName = desc;
            }
            else {
                MMM_ERROR << "Segmentation " << desc << " is unknown!" << std::endl;
                continue;
            }
            float starttime = subsegment.second->getStartTimestep();
            float endtime = subsegment.second->getEndTimestep();
            bool unknownAnnotation = false;
            auto actionAnnotation = subsegment.second->getDerivedAnnotation<ActionLabelAnnotation>();
            if (!actionAnnotation) {
                MMM_ERROR << "No action annotation at " << starttime << " for " << desc << std::endl;
                unknownAnnotation = true;
            }
            else if (actionAnnotation->getLabel()._value == ActionLabel::Idle) {
                MMM_INFO << "Ignoring Idle at " << starttime << " for " << desc << std::endl;
                continue;
            }
            w.increment();
            w.add("Subject", humanMotion->getName());
            w.add("Motion", filename);
            w.add("Nodeset", nodeSetName);
            std::vector<std::string> actions;
            if (parentAction) {
                if (level > 0 && parentAction)
                {
                    w.add("Action" + (level > 1 ? std::to_string(level) : std::string()), parentAction->getLabelStr());
                    actions.push_back(parentAction->getLabelStr());
                }
            }
            if (!unknownAnnotation) {
                w.add("Action" + (level > 0 ? std::to_string(level + 1) : std::string()), actionAnnotation->getLabelStr());
                actions.push_back(actionAnnotation->getLabelStr());
            }
            else {
                w.add("Action" + (level > 0 ? std::to_string(level + 1) : std::string()), "Unknown");
                actions.push_back("Unknown");
            }

            w.add("Start", simox::alg::to_string(starttime));
            w.add("End", simox::alg::to_string(endtime));
            if (!unknownAnnotation) {
                w.add("Main", actionAnnotation->getMainObjectStr());
                w.add("Secondary main", actionAnnotation->getSecondMainObjectStr());
                w.add("Source", actionAnnotation->getSourceStr());
                w.add("Target", actionAnnotation->getTargetStr());
                w.add("Failure", actionAnnotation->getFailureStr());
            }
            else {
                w.add("Main", "");
                w.add("Secondary main", "");
                w.add("Source", "");
                w.add("Target", "");
                w.add("Failure", "");
            }
            w.add("Authors", simox::alg::join(segmentation->getEditors(), ";"));

            WriteFileWrapper ws;
            std::filesystem::path actionOutputFolderPath = outputDir;
            actions.insert(actions.begin(), nodeSetName);
            for (const std::string &actionName : actions) {
                actionOutputFolderPath /= actionName;
                if (!std::filesystem::is_directory(actionOutputFolderPath)
                        || !std::filesystem::exists(actionOutputFolderPath))
                    std::filesystem::create_directory(actionOutputFolderPath); // create folder if not exists
            }
            // TODO write information also in local file
            actionOutputFolderPath /= (humanMotion->getName() + "_-_" + filename + "_-_" + simox::alg::to_string(starttime) + ".csv");

            std::vector<float> timesteps = humanMotion->getTimesteps();
            for (unsigned int i = 0; i < timesteps.size(); i++) {
                float timestep = timesteps[i];
                if (timestep < starttime)
                    continue;
                if (timestep > endtime)
                    break;
                if (i == 0)
                    ws.add("timestep", 0, false);
                else
                    ws.add("timestep", timestep - starttime, false);

                for (const auto &motion : *motions)
                {
                    if (motion->getName() == humanMotion->getName())
                    {
                        motion->initializeModel(mmmModel, timestep);
                        Eigen::Matrix4f rootPose = mmmModel->getGlobalPose();
                        ws.add("MMM_" + motion->getName() + "_ROOT", rootPose, false);
                        Eigen::Matrix4f leftHandPose = mmmModel->getRobotNode("Hand L TCP")->getGlobalPose();
                        ws.add("MMM_" + motion->getName() + "_LH", leftHandPose, false);
                        Eigen::Matrix4f rightHandPose = mmmModel->getRobotNode("Hand R TCP")->getGlobalPose();
                        ws.add("MMM_" + motion->getName() + "_RH", rightHandPose, false);
                    }
                    else if (motion->getName() != "go_pro" && motion->getName() != "kinect_azure_right"
                             && motion->getName() != "kinect_azure_new_right"
                             && motion->getName() != "kinect_azure_new_left"
                             && motion->getName() != "kinect_azure_new_front"
                             && motion->getName() != "kinect_azure_front" && motion->getName() != "kinect_azure_left"){
                        // TODO ignore kinects + gopro
                        Eigen::Matrix4f pose = motion->getRootPose(timestep);
                        ws.add(motion->getName(), pose, false);
                    }
                }
                ws.increment();
            }
            ws.writeCSV(actionOutputFolderPath, true);
            w.add("filepath", simox::fs::make_relative(outputDir, actionOutputFolderPath), false);

            MMM_INFO << "Finished " << (actionAnnotation ? actionAnnotation->getLabel()._to_string() : "") << " at timestep " << starttime << " for " << desc << std::endl;
            saveRecursive(subsegment.second, motions, humanMotion, mmmModel, w, filename, outputDir, maxLevel, level + 1, actionAnnotation);
        }
    }
}

int main(int argc, char *argv[]) {
    MMM_INFO << " --- MMMSegmentwiseCSVExporter --- " << std::endl;
    MMMSegmentwiseCSVExporterConfiguration* configuration = new MMMSegmentwiseCSVExporterConfiguration();
    if (!configuration->processCommandLine(argc, argv)) {
        MMM_ERROR << "Error while processing command line, aborting..." << std::endl;
        return -1;
    }

    std::filesystem::path mmmPath = std::string(MMMTools_BUILD_DIR) + "/../../data/Model/Winter/mmm.xml";
    if (!std::filesystem::exists(mmmPath)) {
        MMM_ERROR << "MMM model does not exist at path " << mmmPath << std::endl;
        return -1;
    }

    if ((!std::filesystem::is_directory(configuration->outputDir) || !std::filesystem::exists(configuration->outputDir)))
        std::filesystem::create_directory(configuration->outputDir); // create folder if not exists

    //std::filesystem::path dataPath = configuration->outputDir / "data";
    //if ((!std::filesystem::is_directory(dataPath) || !std::filesystem::exists(dataPath)))
    //    std::filesystem::create_directory(dataPath); // create folder if not exists

    MMM::LoadModelStrategy::LOAD_MODEL_STRATEGY = MMM::LoadModelStrategy::Strategy::NONE;
    MMM::MotionReaderXMLPtr motionReader(new MMM::MotionReaderXML(true, false));
    std::vector<std::filesystem::path> paths;
    for (const auto &motionFile : configuration->motionFiles) {
        if (std::filesystem::is_directory(motionFile)) {
            std::vector<std::filesystem::path> subPaths = motionReader->getMotionPathsFromDirectory(motionFile);
            paths.insert(paths.end(), subPaths.begin(), subPaths.end());
        }
        else paths.push_back(motionFile);
    }
    for (const auto &path : paths) {
        try {
            MMM::MotionRecordingPtr motions = motionReader->loadMotionRecording(path);
            MotionPtr humanMotion = motions->getReferenceModelMotion();
            std::string filename = path.filename().stem().string();
            std::filesystem::path motionOverviewCSVPath = configuration->outputDir / (humanMotion->getName() + "_-_" + filename + ".csv");
            if (std::filesystem::exists(motionOverviewCSVPath) && !configuration->overwrite) {
                MMM_INFO << "Skipping motion as " << motionOverviewCSVPath << " already exists and overwrite=false" << std::endl;
                continue;
            }

            VirtualRobot::RobotPtr robot = VirtualRobot::RobotIO::loadRobot(mmmPath, VirtualRobot::RobotIO::eStructure);
            VirtualRobot::RobotPtr mmmModel = humanMotion->getModelProcessor()->convertModel(robot);

            //MMM_INFO << "Output file path " << motionOverviewCSVPath << std::endl;
            //unsigned int maxHierarchyLevel = 2;

            //MotionRecordingSegmentPtr segment = motions->getSegment();
            //if (segment) {

                WriteFileWrapper w;
                //w.addHeaderCheck( { "Subject", "Motion", "Nodeset", "Action" }, true);

                //for (unsigned int i = 1; i < maxHierarchyLevel; i++)
                //     w.addHeaderCheck("Action" + std::to_string(i+1), true);
                //w.addHeaderCheck( { "Start", "End", "Main", "Secondary main", "Source", "Target", "Failure",
                //                    "Authors"}, true);

                //saveRecursive(segment, motions, humanMotion, mmmModel, w, filename, dataPath, maxHierarchyLevel);
		    save(motions, humanMotion, mmmModel, w, filename, configuration->outputDir);

                MMM_INFO << "Storing csv file at " << motionOverviewCSVPath << std::endl;
                //w.increment();
		        //w.writeCSV(motionOverviewCSVPath);
            w.writeCSV(motionOverviewCSVPath, true);
            //}
            //else MMM_ERROR << "No segmentation in motion!" << std::endl;
        } catch (MMM::Exception::MMMException& e) {
            MMM_ERROR << "Extracting from " << path << ": "  << e.what() << std::endl;
            continue;
        }
    }
}
