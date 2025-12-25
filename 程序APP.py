import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 设置页面配置
st.set_page_config(
    page_title="PI预测模型",
    page_icon="🏥",
    layout="wide"
)

# 作者和单位信息
AUTHOR_INFO = {
    "author": "石层层",
    "institution": "山东药品食品职业学院"
}

# 加载保存的随机森林模型
model = joblib.load('rf.pkl')

# 特征缩写映射（在后台代码中设置）
feature_abbreviations = {
    "FCTI": "FCTI",
    "Age": "Age",
    "Ser": "Ser",
    "Fra": "Fra",
    "Air": "Air",
    "Com": "Com",
    "PCAT": "PCAT",
    "Mlu": "Mlu"
}

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
   "FCTI": {"type": "numerical", "min": 0, "max": 40, "default": 21, "label": "FCTI总分"},
    "Age": {"type": "numerical", "min": 70, "max": 99, "default": 78, "label": "年龄（岁）"},
    "Ser": {"type": "numerical", "min": 20, "max": 60, "default": 21, "label": "血清白蛋白"},
    "Fra": {"type": "categorical", "options": [0,1,2,3,4,5,6,7,8,9,10,11,12,13], "default": 9, "label": "骨折类型", 
            "option_labels": {0: "颈椎骨折", 1: "胸椎骨折",2: "腰椎骨折", 
                              3: "股骨颈骨折", 4: "股骨粗隆间骨折", 5: "股骨干骨折", 6: "胫腓骨上段骨折",
                              7: "尾骨粉碎性骨折", 8: "骶髂关节脱位", 9: "髋骨骨折", 
                              10: "髌骨粉碎性骨折", 11: "髋关节内骨折", 12: "脆性骨折", 13: "其他"}},
    "Air": {"type": "categorical", "options": [0, 1], "default": 0, "label": "气垫床/充气床垫", "option_labels": {0: "未使用", 1: "使用"}},
    "Com": {"type": "numerical", "min": 0, "max": 8, "default": 2, "label": "合并症数量"},
    "PCAT": {"type": "numerical", "min": 1, "max": 4, "default": 3, "label": "PCAT总分"},
    "Mlu": {"type": "categorical", "options": [0, 1], "default": 0, "label": "多发性骨折", "option_labels": {0: "否", 1: "是"}},
}

# Streamlit 界面
st.title("“医院—家庭—社区”三区联合延续护理模式下的老年骨折卧床患者PI风险预测模型")

# 添加作者信息（在主标题下方）
st.markdown(f"""
<div style='text-align: center; color: #666; margin-top: -10px; margin-bottom: 20px;'>
    开发单位：{AUTHOR_INFO["institution"]} | 作者：{AUTHOR_INFO["author"]}
</div>
""", unsafe_allow_html=True)

# 添加说明文本
st.markdown("""
本应用基于机器学习模型预测在“医院—家庭—社区”三区联合延续护理模式下的老年骨折卧床患者PI风险。
请在下方的表单中输入患者的临床指标，然后点击"开始预测"按钮。
""")

# 动态生成输入项
st.header("请输入患者临床指标:")
feature_values = []

# 创建两列布局，使界面更紧凑
col1, col2 = st.columns(2)

features_list = list(feature_ranges.keys())
half_point = len(features_list) // 2

for i, feature in enumerate(features_list):
    properties = feature_ranges[feature]
    
    # 根据位置选择列
    if i < half_point:
        with col1:
            if properties["type"] == "numerical":
                value = st.number_input(
                    label=f"{properties['label']}",
                    min_value=float(properties["min"]),
                    max_value=float(properties["max"]),
                    value=float(properties["default"]),
                    help=f"范围: {properties['min']} - {properties['max']}"
                )
            elif properties["type"] == "categorical":
                # 对于分类变量，使用选择框并显示中文标签
                option_labels = properties.get("option_labels", {k: str(k) for k in properties["options"]})
                selected_label = st.selectbox(
                    label=f"{properties['label']}",
                    options=properties["options"],
                    format_func=lambda x: option_labels[x],
                    index=properties["options"].index(properties["default"])
                )
                value = selected_label
            feature_values.append(value)
    else:
        with col2:
            if properties["type"] == "numerical":
                value = st.number_input(
                    label=f"{properties['label']}",
                    min_value=float(properties["min"]),
                    max_value=float(properties["max"]),
                    value=float(properties["default"]),
                    help=f"范围: {properties['min']} - {properties['max']}"
                )
            elif properties["type"] == "categorical":
                option_labels = properties.get("option_labels", {k: str(k) for k in properties["options"]})
                selected_label = st.selectbox(
                    label=f"{properties['label']}",
                    options=properties["options"],
                    format_func=lambda x: option_labels[x],
                    index=properties["options"].index(properties["default"])
                )
                value = selected_label
            feature_values.append(value)

# 添加一个分隔线
st.markdown("---")

# 预测与 SHAP 可视化
if st.button("开始预测", type="primary"):
    # 显示加载指示器
    with st.spinner('模型正在计算中，请稍候...'):
        # 转换为模型输入格式
        features = np.array([feature_values])

        # 模型预测
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]

        # 提取预测的类别概率
        probability = predicted_proba[predicted_class] * 100

    # 显示预测结果
    st.subheader("预测结果")
    
    # 使用进度条和指标显示概率
    st.metric(label="PI发生概率", value=f"{probability:.2f}%")
    st.progress(int(probability))
    
    # 添加风险等级解读
    if probability < 20:
        risk_level = "低风险"
        color = "green"
    elif probability < 50:
        risk_level = "中风险"
        color = "orange"
    else:
        risk_level = "高风险"
        color = "red"
    
    st.markdown(f"<h4 style='color: {color};'>风险等级: {risk_level}</h4>", unsafe_allow_html=True)

    # 计算 SHAP 值
    with st.spinner('正在生成模型解释图...'):
        explainer = shap.TreeExplainer(model)
        
        # 创建用于SHAP的DataFrame，使用缩写作为列名
        shap_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())
        shap_df.columns = [feature_abbreviations[col] for col in shap_df.columns]
        
        shap_values = explainer.shap_values(shap_df)

        # 生成 SHAP 力图
        class_index = predicted_class  # 当前预测类别
        plt.figure(figsize=(10, 4))
        shap_plot = shap.force_plot(
            explainer.expected_value[class_index],
            shap_values[:,:,class_index],
            shap_df,  # 使用带有缩写的DataFrame
            matplotlib=True,
            show=False
        )
        
        # 保存并显示 SHAP 图
        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300, 
                   facecolor='white', edgecolor='none')
        plt.close()

    # 显示SHAP解释图
    st.subheader("模型解释")
    st.markdown("下图显示了各个特征变量对预测结果的贡献程度：")
    st.image("shap_force_plot.png")
    
    # 添加特征缩写说明
    with st.expander("特征缩写说明"):
        st.markdown("| 缩写 | 全称 | 描述 |")
        st.markdown("|------|------|------|")
        st.markdown("| Age | Age | 年龄 |")
        st.markdown("| Cog | Cognize | 照护者技能 |")
        st.markdown("| Com | Complications | 合并症数量 |")
        st.markdown("| CG | CareGuidance | 照护指导 |")
        st.markdown("| ACB | AirCushionBed | 气垫床/充气床垫 |")
        st.markdown("| RC | ResourceCoordination | 资源协调与支持 |")
        st.markdown("| PF | PelvicFracture | 盆骨骨折 |")
        st.markdown("| SF | SpecialFracture | 特殊骨折 |")
    
    # 添加图例说明
    with st.expander("如何解读此图"):
        st.markdown("""
        - **红色箭头**：增加PI风险的因素
        - **蓝色箭头**：降低PI风险的因素  
        - **箭头长度**：表示该因素影响程度的大小
        - **基准值**：模型在训练数据上的平均预测值
        - **输出值**：当前患者的预测值
        """)

# 添加侧边栏信息
with st.sidebar:
    st.header("关于本应用")
    st.markdown(f"""
    ### 开发信息
    - **开发单位**: {AUTHOR_INFO["institution"]}
    - **作者**: {AUTHOR_INFO["author"]}
    
    ### 模型信息
    - **算法**: XGBoost
    - **预测目标**: 压力性损伤(PI)
    - **应用场景**: 临床风险评估
    
    ### 使用说明
    1. 在右侧表单中输入患者临床指标
    2. 点击"开始预测"按钮
    3. 查看预测结果和模型解释
    
    ### 注意事项
    - 本工具仅供临床参考
    - 实际诊疗请结合临床判断
    - 如有疑问请咨询专业医师
    """)

# 添加页脚
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray;'>
        临床决策支持工具 • {AUTHOR_INFO["institution"]} • {AUTHOR_INFO["author"]} • 仅供参考
    </div>
    """, 
    unsafe_allow_html=True
)