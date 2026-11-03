// SPDX-License-Identifier: MIT
#include <OgreHighLevelGpuProgramManager.h>
#include <OgreHighLevelGpuProgram.h>
#include <OgreManualObject.h>
#include <OgreMaterialManager.h>
#include <OgreTechnique.h>
#include <OgrePass.h>
#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/properties/float_property.hpp>
#include <rviz_common/ros_topic_display.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace gcopter {
// Viewer for the exterior map triangles published by direct_corridor_demo.
// Alpha is a material uniform: dragging the slider never resends the mesh.
class GhostMapDisplay final
    : public rviz_common::RosTopicDisplay<visualization_msgs::msg::MarkerArray> {
 public:
  GhostMapDisplay() {
    opacity_ = new rviz_common::properties::FloatProperty(
        "Opacity", .16, "Opacity of the translucent obstacle surface.", this);
    opacity_->setMin(0);
    opacity_->setMax(1);
    connect(opacity_, &rviz_common::properties::Property::changed,
            this, [this] { updateOpacity(); });
  }
  ~GhostMapDisplay() override {
    if (mesh_) scene_manager_->destroyManualObject(mesh_);
    for (const auto& material : materials_)
      Ogre::MaterialManager::getSingleton().remove(material->getName());
  }
  void onInitialize() override {
    RTDClass::onInitialize();
    mesh_ = scene_manager_->createManualObject();
    scene_node_->attachObject(mesh_);
    createProgram("gcopter/ghost/vertex", Ogre::GPT_VERTEX_PROGRAM,
        "#version 120\nattribute vec4 vertex; uniform mat4 worldViewProj;\n"
        "void main() { gl_Position = worldViewProj * vertex; }\n");
    createProgram("gcopter/ghost/fragment", Ogre::GPT_FRAGMENT_PROGRAM,
        "#version 120\nuniform vec4 tint;\n"
        "void main() { gl_FragColor = tint; }\n");
  }
  void reset() override {
    RTDClass::reset();
    if (mesh_) mesh_->clear();
  }

 protected:
  void processMessage(visualization_msgs::msg::MarkerArray::ConstSharedPtr message) override {
    QMetaObject::invokeMethod(this, [this, message] { drawMap(*message); }, Qt::QueuedConnection);
  }

 private:
  static void createProgram(const std::string& name, Ogre::GpuProgramType type,
                            const std::string& source) {
    auto& manager = Ogre::HighLevelGpuProgramManager::getSingleton();
    if (manager.resourceExists(name)) return;
    auto program = manager.createProgram(name, Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
                                         "glsl", type);
    program->setSource(source);
    program->load();
  }
  void drawMap(const visualization_msgs::msg::MarkerArray& message) {
    if (!isEnabled()) return;
    mesh_->clear();
    std::size_t slot = 0;
    for (const auto& marker : message.markers) {
      if (marker.type != marker.TRIANGLE_LIST || marker.action != marker.ADD || marker.points.empty())
        continue;
      Ogre::Vector3 position;
      Ogre::Quaternion orientation;
      if (!context_->getFrameManager()->getTransform(marker.header, position, orientation)) {
        setStatus(rviz_common::properties::StatusProperty::Error, "Map", "Map frame is unavailable");
        return;
      }
      scene_node_->setPosition(position);
      scene_node_->setOrientation(orientation);
      if (slot == materials_.size()) {
        const std::string name = "gcopter/ghost/material/" +
            std::to_string(reinterpret_cast<std::uintptr_t>(this)) + "/" + std::to_string(slot);
        auto material = Ogre::MaterialManager::getSingleton().create(
            name, Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
        auto* pass = material->getTechnique(0)->getPass(0);
        pass->setVertexProgram("gcopter/ghost/vertex");
        pass->getVertexProgramParameters()->setNamedAutoConstant(
            "worldViewProj", Ogre::GpuProgramParameters::ACT_WORLDVIEWPROJ_MATRIX);
        pass->setFragmentProgram("gcopter/ghost/fragment");
        pass->setLightingEnabled(false);
        pass->setSceneBlending(Ogre::SBT_TRANSPARENT_ALPHA);
        pass->setDepthWriteEnabled(false);
        pass->setDepthCheckEnabled(true);
        pass->setCullingMode(Ogre::CULL_CLOCKWISE);
        materials_.push_back(material);
        // A subdued blue-grey tint; the existing face shading gives depth.
        tints_.emplace_back(marker.color.r * .82f, marker.color.g * .88f,
                            marker.color.b * .94f, opacity_->getFloat());
      }
      mesh_->begin(materials_[slot]->getName(), Ogre::RenderOperation::OT_TRIANGLE_LIST);
      for (const auto& p : marker.points) mesh_->position(p.x, p.y, p.z);
      mesh_->end();
      ++slot;
    }
    updateOpacity();
    setStatus(rviz_common::properties::StatusProperty::Ok, "Map", "Translucent exterior surface");
  }
  void updateOpacity() {
    for (std::size_t i = 0; i < materials_.size(); ++i) {
      tints_[i].w = opacity_->getFloat();
      materials_[i]->getTechnique(0)->getPass(0)->getFragmentProgramParameters()
          ->setNamedConstant("tint", tints_[i]);
    }
    if (context_) context_->queueRender();
  }
  rviz_common::properties::FloatProperty* opacity_;
  Ogre::ManualObject* mesh_ = nullptr;
  std::vector<Ogre::MaterialPtr> materials_;
  std::vector<Ogre::Vector4> tints_;
};
}  // namespace gcopter
PLUGINLIB_EXPORT_CLASS(gcopter::GhostMapDisplay, rviz_common::Display)
