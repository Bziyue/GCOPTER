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
#include <QScrollArea>
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
    setMinimumWidth(320);
    setMaximumWidth(390);
    setStyleSheet(
        "#CorridorPanel { background:#f7f9fc; }"
        "#CorridorPanel QLabel { color:#243449; background:transparent; }"
        "#CorridorPanel QFrame#card { background:white; border:1px solid #dce4ed; border-radius:10px; }"
        "#CorridorPanel QCheckBox { color:#33465c; padding:2px 0; spacing:9px; }"
        "#CorridorPanel QPushButton { color:#126eaa; background:#e9f3fb; border:1px solid #bcd9ed;"
        "border-radius:7px; padding:10px; font-weight:600; }"
        "#CorridorPanel QPushButton:hover { background:#d8ecfa; }");
    auto* outer = new QVBoxLayout(this);outer->setContentsMargins(0,0,0,0);
    auto* scroll = new QScrollArea;scroll->setWidgetResizable(true);scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);scroll->setFrameShape(QFrame::NoFrame);
    auto* content = new QWidget;scroll->setWidget(content);outer->addWidget(scroll);
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(18, 20, 18, 18);
    layout->setSpacing(8);
    kicker_ = label("GCOPTER  /  C3 · C²", 10, "#71839a");
    layout->addWidget(kicker_);
    title_ = label("走廊与轨迹对比", 20, "#172b43", true);
    subtitle_ = label("同地图 · 同路径 · 同轨迹初值", 10, "#71839a");
    layout->addWidget(title_);
    layout->addWidget(subtitle_);
    phase_ = label("正在连接地图…", 12, "#287eac", true);
    phase_->setWordWrap(true);
    layout->addWidget(phase_);

    auto* card = new QFrame;
    card->setObjectName("card");
    auto* metrics = new QVBoxLayout(card);
    metrics->setContentsMargins(16, 16, 16, 16);
    metrics->setSpacing(10);
    auto* times=new QGridLayout;
    corridor_name_=label("C3 走廊",10,"#b31f33");
    times->addWidget(corridor_name_,0,0);
    times->addWidget(label("FIRI 走廊",10,"#126eaa"),0,1);
    duration_=label("—",21,"#b31f33",true);
    firi_duration_=label("—",21,"#126eaa",true);
    times->addWidget(duration_,1,0);times->addWidget(firi_duration_,1,1);
    metrics->addLayout(times);
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

    optimize_button_=new QPushButton("优化两条轨迹");optimize_button_->setEnabled(false);
    optimize_button_->setToolTip("两种走廊共用五次最小 jerk 轨迹、动力学约束和初值，仅位置约束不同");
    layout->addWidget(optimize_button_);
    connect(optimize_button_,&QPushButton::clicked,this,[this]{
      optimize_button_->setEnabled(false);optimize_button_->setText("正在优化…");
      optimize_->publish(std_msgs::msg::Empty());
    });
    firi_result_=label("FIRI / GCOPTER：等待优化",10,"#126eaa");
    c3_result_=label("C3：等待优化",10,"#b31f55");
    for(auto* result:{firi_result_,c3_result_}){result->setWordWrap(true);layout->addWidget(result);}
    layout->addWidget(label("显示图层", 11, "#71839a", true));
    addLayer(layout, "C3 走廊", "Corridor surface", "#f03d40");
    addLayer(layout, "FIRI 凸多面体", "FIRI corridor", "#147acc");
    addLayer(layout, "C3 轨迹", "C3 trajectory", "#cc0f40");
    addLayer(layout, "FIRI / GCOPTER 轨迹", "FIRI trajectory", "#0640ad");
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
    optimize_ = node->create_publisher<std_msgs::msg::Empty>("/direct_demo/optimize",1);
    comparison_ = node->create_subscription<std_msgs::msg::String>("/direct_demo/comparison",qos,
      [this](std_msgs::msg::String::ConstSharedPtr m){
        const QByteArray data=QByteArray::fromStdString(m->data);
        QMetaObject::invokeMethod(this,[this,data]{showComparison(data);},Qt::QueuedConnection);
      });
    reset_ = node->create_publisher<std_msgs::msg::Empty>("/direct_demo/reset", 1);
    QTimer::singleShot(0, this, [this] {
      applyOpacity();auto* group=getDisplayContext()->getRootDisplayGroup();
      for(const auto& layer:layers_)for(int i=0;i<group->numDisplays();++i)
        if(group->getDisplayAt(i)->getName()==layer.first){const QSignalBlocker block(layer.second);layer.second->setChecked(group->getDisplayAt(i)->isEnabled());}
    });
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
    auto* container = new QWidget;
    auto* row = new QHBoxLayout(container);
    row->setContentsMargins(0,0,0,0);
    auto* swatch = new QLabel;
    swatch->setFixedSize(9, 9);
    swatch->setStyleSheet(QString("background:%1; border-radius:4px;").arg(color));
    auto* toggle = new QCheckBox(title);
    toggle->setChecked(true);
    layers_.push_back({display,toggle});
    row->addWidget(swatch);
    row->addWidget(toggle);
    layout->addWidget(container);
    connect(toggle, &QCheckBox::toggled, this, [this, display](bool enabled) {
      auto* group = getDisplayContext()->getRootDisplayGroup();
      for (int i = 0; i < group->numDisplays(); ++i)
        if (group->getDisplayAt(i)->getName() == display ||
            (display=="Corridor surface"&&(group->getDisplayAt(i)->getName()=="Cross sections"||group->getDisplayAt(i)->getName()=="Centerline")))
          group->getDisplayAt(i)->setEnabled(enabled);
      if(display=="Corridor surface")for(const auto& layer:layers_)
        if(layer.first=="Cross sections"||layer.first=="Centerline"){const QSignalBlocker block(layer.second);layer.second->setChecked(enabled);}
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
    else { caption = "走廊未生成，请重新选点"; color = "#ba5347"; }
    phase_->setText(caption);
    phase_->setStyleSheet("color:" + color + ";");
    phase_->setToolTip(text);
  }
  void showMetrics(const QByteArray& data) {
    if (data.isEmpty()) {
      for (auto* value : {duration_, search_, length_, radius_, firi_duration_}) value->setText("—");
      return;
    }
    const auto m = QJsonDocument::fromJson(data).object();
    firi_duration_->setText(m["firi_ready"].toBool()?QString::number(m["firi_ms"].toDouble(),'f',1)+" ms":"未生成");
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
  void showComparison(const QByteArray& data) {
    auto m=QJsonDocument::fromJson(data).object();QString state=m["state"].toString();
    if(m.contains("shape_mode")) {
      const bool h3=m["shape_mode"].toString()=="h3";
      kicker_->setText(h3?"GCOPTER  /  H3 · C¹":"GCOPTER  /  C3 · C²");
      title_->setText(h3?"H3 走廊生成":"走廊与轨迹对比");
      subtitle_->setText(h3?"同地图 · 同路径 · 与 FIRI 对比":"同地图 · 同路径 · 同轨迹初值");
      corridor_name_->setText(h3?"H3 走廊":"C3 走廊");
      optimize_button_->setVisible(!h3);
      firi_result_->setVisible(!h3);c3_result_->setVisible(!h3);
      for(const auto& layer:layers_) {
        if(layer.first=="Corridor surface")layer.second->setText(h3?"H3 走廊":"C3 走廊");
        if(layer.first=="C3 trajectory"||layer.first=="FIRI trajectory")
          layer.second->parentWidget()->setVisible(!h3);
      }
    }
    optimize_button_->setEnabled(m["ready"].toBool()&&!m["busy"].toBool());
    optimize_button_->setText(m["busy"].toBool()?"正在优化…":"优化两条轨迹");
    if(state=="DONE") {
      auto show=[](QLabel* label,const QString& name,const QJsonObject& result) {
        bool ok=result["sampled_feasible"].toBool();
        label->setText(QString("%1：%2 ms · %3\n%4 m / %5 s · 越界 %6 mm")
          .arg(name).arg(result["optimize_ms"].toDouble(),0,'f',1).arg(ok?"采样通过":"待检查")
          .arg(result["length"].toDouble(),0,'f',1).arg(result["duration"].toDouble(),0,'f',1)
          .arg(1000*result["max_violation"].toDouble(),0,'f',1));
        label->setToolTip(QString("求解状态 %1；迭代 %2；jerk %3；障碍穿透 %4 mm")
          .arg(result["status"].toInt()).arg(result["iterations"].toInt())
          .arg(result["jerk"].toDouble()).arg(1000*result["penetration"].toDouble()));
      };
      show(firi_result_,"FIRI / GCOPTER",m["firi"].toObject());show(c3_result_,"C3",m["c3"].toObject());
    } else {
      QString text=state=="RUNNING"?"正在串行求解":state=="ERROR"?"优化失败，可重试":"等待优化";
      firi_result_->setText("FIRI / GCOPTER："+text);c3_result_->setText("C3："+text);
    }
  }
  std::vector<std::pair<QString,QCheckBox*>> layers_;
  QPushButton* optimize_button_;
  QLabel *firi_duration_,*firi_result_,*c3_result_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr optimize_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr comparison_;
  QLabel *phase_, *duration_, *search_, *length_, *radius_, *opacity_value_;
  QLabel *kicker_, *title_, *subtitle_, *corridor_name_;
  QSlider* transparency_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr status_, metrics_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr reset_;
};
}  // namespace gcopter
PLUGINLIB_EXPORT_CLASS(gcopter::CorridorPanel, rviz_common::Panel)
