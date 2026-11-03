// SPDX-License-Identifier: MIT
#include <QCheckBox>
#include <QFrame>
#include <QGridLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QPainter>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSlider>
#include <QTimer>
#include <QVBoxLayout>
#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/display_group.hpp>
#include <rviz_common/panel.hpp>
#include <rviz_common/render_panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <rviz_common/view_manager.hpp>
#include <rviz_rendering/render_window.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/string.hpp>

namespace gcopter {
class CorridorPanel final : public rviz_common::Panel {
 public:
  explicit CorridorPanel(QWidget* parent = nullptr) : Panel(parent) {
    setObjectName("CorridorPanel");
    setMinimumWidth(285);
    setMaximumWidth(390);
    setStyleSheet(
        "#CorridorPanel { background:#f7f9fc; }"
        "#CorridorPanel QLabel { color:#243449; background:transparent; }"
        "#CorridorPanel QFrame#card { background:white; border:1px solid #dce4ed; border-radius:10px; }"
        "#CorridorPanel QCheckBox { color:#33465c; padding:5px 0; spacing:9px; }"
        "#CorridorPanel QPushButton { color:#126eaa; background:#e9f3fb; border:1px solid #bcd9ed;"
        "border-radius:7px; padding:10px; font-weight:600; }"
        "#CorridorPanel QPushButton:hover { background:#d8ecfa; }");
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(18, 20, 18, 18);
    layout->setSpacing(15);
    auto* kicker = label("GCOPTER  /  C²", 10, "#71839a");
    layout->addWidget(kicker);
    layout->addWidget(label("可微走廊", 23, "#172b43", true));
    layout->addWidget(label("从路径到连续自由空间", 10, "#71839a"));
    phase_ = label("正在连接地图…", 12, "#287eac", true);
    phase_->setWordWrap(true);
    layout->addWidget(phase_);

    auto* card = new QFrame;
    card->setObjectName("card");
    auto* metrics = new QVBoxLayout(card);
    metrics->setContentsMargins(16, 16, 16, 16);
    metrics->setSpacing(10);
    metrics->addWidget(label("走廊生成", 10, "#71839a"));
    duration_ = label("—", 29, "#126eaa", true);
    metrics->addWidget(duration_);
    auto* detail = new QGridLayout;
    detail->setHorizontalSpacing(18);
    detail->setVerticalSpacing(7);
    search_ = label("—", 11, "#243449", true);
    length_ = label("—", 11, "#243449", true);
    radius_ = label("—", 11, "#243449", true);
    detail->addWidget(label("路径搜索", 10, "#71839a"), 0, 0);
    detail->addWidget(search_, 0, 1);
    detail->addWidget(label("路径长度", 10, "#71839a"), 1, 0);
    detail->addWidget(length_, 1, 1);
    detail->addWidget(label("最小半轴", 10, "#71839a"), 2, 0);
    detail->addWidget(radius_, 2, 1);
    metrics->addLayout(detail);
    layout->addWidget(card);

    layout->addWidget(label("显示图层", 11, "#71839a", true));
    addLayer(layout, "走廊曲面", "Corridor surface", "#f03d40");
    addLayer(layout, "稀疏截面", "Cross sections", "#b31f33");
    addLayer(layout, "走廊中心线", "Centerline", "#8b192a");
    addLayer(layout, "搜索路径", "Seed path", "#d69535");
    addLayer(layout, "障碍物表面", "Environment", "#9eafc0");
    auto* opacity_row = new QHBoxLayout;
    opacity_row->addWidget(label("障碍物透明度", 10, "#71839a"));
    opacity_value_ = label("84%", 10, "#287eac", true);
    opacity_row->addStretch();
    opacity_row->addWidget(opacity_value_);
    layout->addLayout(opacity_row);
    transparency_ = new QSlider(Qt::Horizontal);
    transparency_->setRange(0, 100);
    transparency_->setValue(84);
    transparency_->setToolTip("向右拖动，让障碍物更透明");
    layout->addWidget(transparency_);
    connect(transparency_, &QSlider::valueChanged, this, [this](int value) {
      opacity_value_->setText(QString::number(value) + "%");
      applyOpacity();
    });
    auto* instructions = label("使用顶部 2D Goal Pose\n① 点选起点  ② 点选终点\n再次点选可开始一条新路径", 10, "#71839a");
    instructions->setWordWrap(true);
    layout->addWidget(instructions);
    auto* reset = new QPushButton("重新选点");
    layout->addWidget(reset);
    connect(reset, &QPushButton::clicked, this, [this] {
      reset_->publish(std_msgs::msg::Empty());
    });
    layout->addStretch();
  }

  void save(rviz_common::Config config) const override {
    Panel::save(config);
    config.mapSetValue("Obstacle Transparency", transparency_->value());
  }
  void load(const rviz_common::Config& config) override {
    Panel::load(config);
    int value = 84;
    config.mapGetInt("Obstacle Transparency", &value);
    const QSignalBlocker blocker(transparency_);
    transparency_->setValue(value);
    opacity_value_->setText(QString::number(transparency_->value()) + "%");
  }

  void onInitialize() override {
    auto node = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();
    const auto qos = rclcpp::QoS(1).reliable().transient_local();
    reset_ = node->create_publisher<std_msgs::msg::Empty>("/direct_demo/reset", 1);
    QTimer::singleShot(0, this, [this] { applyOpacity(); });
    status_ = node->create_subscription<std_msgs::msg::String>(
        "/direct_demo/status", qos, [this](std_msgs::msg::String::ConstSharedPtr m) {
          const QString text = QString::fromStdString(m->data);
          QMetaObject::invokeMethod(this, [this, text] { showStatus(text); }, Qt::QueuedConnection);
        });
    metrics_ = node->create_subscription<std_msgs::msg::String>(
        "/direct_demo/metrics", qos, [this](std_msgs::msg::String::ConstSharedPtr m) {
          const QByteArray data = QByteArray::fromStdString(m->data);
          QMetaObject::invokeMethod(this, [this, data] { showMetrics(data); }, Qt::QueuedConnection);
        });
    // Optional visual QA export. Capture Ogre directly; desktop screenshots
    // of XWayland OpenGL windows may otherwise contain a black render area.
    const QString preview = qEnvironmentVariable("GCOPTER_RVIZ_PREVIEW");
    if (!preview.isEmpty()) QTimer::singleShot(3500, this, [this, preview] {
      auto* panel = getDisplayContext()->getViewManager()->getRenderPanel();
      panel->getRenderWindow()->captureScreenShot((preview + ".scene.png").toStdString());
      QPixmap snapshot = window()->grab();
      QPainter painter(&snapshot);
      painter.drawPixmap(QRect(panel->mapTo(window(), QPoint()), panel->size()),
                         QPixmap(preview + ".scene.png"));
      painter.end();
      snapshot.save(preview);
    });
  }

 private:
  void applyOpacity() {
    auto* group = getDisplayContext()->getRootDisplayGroup();
    for (int i = 0; i < group->numDisplays(); ++i)
      if (group->getDisplayAt(i)->getName() == "Environment")
        group->getDisplayAt(i)->subProp("Opacity")->setValue(1. - transparency_->value() / 100.);
  }
  static QLabel* label(const QString& text, int size, const char* color, bool bold = false) {
    auto* widget = new QLabel(text);
    QFont font = widget->font();
    font.setPointSize(size);
    font.setBold(bold);
    widget->setFont(font);
    widget->setStyleSheet(QString("color:%1;").arg(color));
    return widget;
  }
  void addLayer(QVBoxLayout* layout, const QString& title, const QString& display, const char* color) {
    auto* row = new QHBoxLayout;
    auto* swatch = new QLabel;
    swatch->setFixedSize(9, 9);
    swatch->setStyleSheet(QString("background:%1; border-radius:4px;").arg(color));
    auto* toggle = new QCheckBox(title);
    toggle->setChecked(true);
    row->addWidget(swatch);
    row->addWidget(toggle);
    layout->addLayout(row);
    connect(toggle, &QCheckBox::toggled, this, [this, display](bool enabled) {
      auto* group = getDisplayContext()->getRootDisplayGroup();
      for (int i = 0; i < group->numDisplays(); ++i)
        if (group->getDisplayAt(i)->getName() == display)
          group->getDisplayAt(i)->setEnabled(enabled);
    });
  }
  void showStatus(const QString& text) {
    QString caption, color = "#287eac";
    if (text.startsWith("WAITING_FOR_MAP")) caption = "正在加载地图…";
    else if (text.startsWith("WAITING_FOR_START")) caption = "① 请选择起点";
    else if (text.startsWith("WAITING_FOR_GOAL")) caption = "② 请选择终点";
    else if (text.startsWith("SEARCHING")) caption = "正在搜索路径…";
    else if (text.startsWith("GENERATING")) caption = "正在构造走廊…";
    else if (text.startsWith("PASS")) { caption = "✓ 走廊已生成"; color = "#188765"; }
    else if (text.startsWith("INVALID_POINT")) { caption = "请选择有足够净空的位置"; color = "#b66c27"; }
    else { caption = "本次未生成，请重新选点"; color = "#ba5347"; }
    phase_->setText(caption);
    phase_->setStyleSheet("color:" + color + ";");
    phase_->setToolTip(text);
  }
  void showMetrics(const QByteArray& data) {
    if (data.isEmpty()) {
      for (auto* value : {duration_, search_, length_, radius_}) value->setText("—");
      return;
    }
    const auto m = QJsonDocument::fromJson(data).object();
    search_->setText(QString::number(m["search_ms"].toDouble(), 'f', 0) + " ms");
    length_->setText(QString::number(m["seed_length"].toDouble(), 'f', 1) + " m");
    if (m["status"].toString() == "PASS") {
      duration_->setText(QString::number(m["corridor_ms"].toDouble(), 'f', 1) + " ms");
      radius_->setText(QString::number(m["min_axis"].toDouble(), 'f', 2) + " m");
    } else {
      duration_->setText("—");
      radius_->setText("—");
    }
  }
  QLabel *phase_, *duration_, *search_, *length_, *radius_, *opacity_value_;
  QSlider* transparency_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr status_, metrics_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr reset_;
};
}  // namespace gcopter
PLUGINLIB_EXPORT_CLASS(gcopter::CorridorPanel, rviz_common::Panel)
