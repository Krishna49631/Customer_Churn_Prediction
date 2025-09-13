import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os

# Lightweight imports - disable heavy libraries for now
use_heavy_libs = False
if use_heavy_libs:
    import shap
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from lifelines import KaplanMeierFitter

# Load trained model WITH CACHING
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

# Load dataset for Insights Tab WITH CACHING
@st.cache_data
def load_data():
    data = pd.read_csv("Churn.csv")
    return data.sample(n=1000) if len(data) > 1000 else data  # Sample for faster loading

# Load model and data
model = load_model()
df = load_data()

# Streamlit Config
st.set_page_config(
    page_title="Customer Churn Prediction", 
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown("<h1 style='text-align: center; color: violet;'>📊 Customer Churn Prediction Dashboard</h1>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Prediction", 
    "📈 Insights", 
    "📊 Explainability", 
    "📉 Business Metrics", 
    "⏳ Time-to-Churn",
    "🤝 Network Analysis"
])

# ------------------- TAB 1: PREDICTION -------------------
with tab1:
    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.number_input("Tenure (in months)", min_value=0, max_value=72, step=1, value=12)
        contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    with col2:
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, step=1.0, value=65.0)
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col3:
        total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, step=1.0, value=780.0)
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    if st.button("🔍 Predict", type="primary"):
        try:
            # Raw Input
            input_data = pd.DataFrame([{
                "tenure": tenure,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "Contract": contract_type,
                "InternetService": internet_service,
                "PaymentMethod": payment_method
            }])

            # Encode categorical vars same as training
            input_encoded = pd.get_dummies(input_data)

            # Align with model features - ERROR HANDLING
            try:
                # Try to get feature names from trained model
                model_features = model.feature_names_in_
            except AttributeError:
                # Agar model mein feature names nahi hain toh manually define karo
                model_features = [
                    'tenure', 'MonthlyCharges', 'TotalCharges',
                    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
                    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
                    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
                    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
                ]

            input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

            # Predict
            prediction = model.predict(input_encoded)[0]
            prob = model.predict_proba(input_encoded)[0][1]

            # Gauge Meter
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Churn Risk (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if prob > 0.5 else "green"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 100], 'color': "lightcoral"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

            # Result
            if prediction == 1:
                st.error(f"⚠️ Customer is LIKELY to churn (Probability: {prob:.2f})")
            else:
                st.success(f"✅ Customer is NOT likely to churn (Probability: {prob:.2f})")

            # Lightweight explanation instead of SHAP
            st.write("### 🔍 Top Factors Influencing Prediction")
            
            # Manual feature importance based on input values
            factors = []
            if tenure < 6:
                factors.append("Low tenure (<6 months) - higher churn risk")
            if contract_type == "Month-to-month":
                factors.append("Month-to-month contract - higher churn risk")
            if internet_service == "Fiber optic":
                factors.append("Fiber optic service - check satisfaction")
            if payment_method == "Electronic check":
                factors.append("Electronic check payment - higher churn risk")
            if monthly_charges > 70:
                factors.append("High monthly charges - may cause churn")
                
            if factors:
                for factor in factors:
                    st.write(f"• {factor}")
            else:
                st.info("No strong risk factors identified")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ------------------- TAB 2: INSIGHTS -------------------
with tab2:
    st.subheader("Customer Churn Insights")
    
    # Convert Churn to numeric for analysis
    df_numeric = df.copy()
    df_numeric['Churn'] = df_numeric['Churn'].map({"Yes": 1, "No": 0})
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Overall churn rate
        churn_rate = df_numeric['Churn'].mean()
        st.metric("Overall Churn Rate", f"{churn_rate:.2%}")
        
        # Churn by contract type
        contract_churn = df_numeric.groupby('Contract')['Churn'].mean()
        fig = px.bar(x=contract_churn.index, y=contract_churn.values, 
                    title="Churn Rate by Contract Type",
                    labels={'x': 'Contract Type', 'y': 'Churn Rate'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average tenure
        avg_tenure = df_numeric['tenure'].mean()
        st.metric("Average Tenure", f"{avg_tenure:.1f} months")
        
        # Churn by internet service
        internet_churn = df_numeric.groupby('InternetService')['Churn'].mean()
        fig = px.pie(values=internet_churn.values, names=internet_churn.index,
                    title="Churn Distribution by Internet Service")
        st.plotly_chart(fig, use_container_width=True)
    
    # Tenure distribution
    fig = px.histogram(df, x='tenure', color='Churn', 
                      title='Tenure Distribution by Churn Status',
                      barmode='overlay', opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

# ------------------- TAB 3: EXPLAINABILITY -------------------
with tab3:
    st.subheader("Model Explainability")
    
    st.info("""
    **Feature Importance Explanation**
    
    Based on typical churn models, the most important factors are usually:
    - Tenure (how long customer has been with the company)
    - Contract type (month-to-month vs longer contracts)
    - Monthly charges
    - Internet service type
    - Payment method
    """)
    
    # Manual feature importance display
    features = [
        {'feature': 'Tenure', 'importance': 0.25, 'impact': 'Negative'},
        {'feature': 'Contract Type', 'importance': 0.20, 'impact': 'Negative'},
        {'feature': 'Monthly Charges', 'importance': 0.15, 'impact': 'Positive'},
        {'feature': 'Internet Service', 'importance': 0.15, 'impact': 'Mixed'},
        {'feature': 'Payment Method', 'importance': 0.10, 'impact': 'Mixed'},
        {'feature': 'Total Charges', 'importance': 0.08, 'impact': 'Negative'},
        {'feature': 'Additional Services', 'importance': 0.07, 'impact': 'Negative'}
    ]
    
    importance_df = pd.DataFrame(features)
    fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                title='Typical Feature Importance in Churn Models',
                color='impact', color_discrete_map={
                    'Negative': 'green', 
                    'Positive': 'red', 
                    'Mixed': 'orange'
                })
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("""
    **Impact Explanation:**
    - **Negative**: Higher values reduce churn probability
    - **Positive**: Higher values increase churn probability  
    - **Mixed**: Effect depends on other factors
    """)

# ------------------- TAB 4: BUSINESS METRICS -------------------
with tab4:
    st.subheader("Business Impact Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Customers", len(df))
        
    with col2:
        churn_count = len(df[df['Churn'] == 'Yes'])
        st.metric("Churned Customers", churn_count)
        
    with col3:
        avg_revenue = df['MonthlyCharges'].mean() if 'MonthlyCharges' in df.columns else 0
        st.metric("Avg Monthly Revenue", f"${avg_revenue:.2f}")
    
    # Simple revenue impact analysis
    st.write("### Revenue Impact Estimate")
    
    if 'MonthlyCharges' in df.columns:
        monthly_revenue_loss = df[df['Churn'] == 'Yes']['MonthlyCharges'].sum()
        annual_revenue_loss = monthly_revenue_loss * 12
        
        st.write(f"**Monthly revenue loss from churn:** ${monthly_revenue_loss:,.2f}")
        st.write(f"**Annual revenue loss from churn:** ${annual_revenue_loss:,.2f}")
        
        # Retention benefit
        st.write("""
        **Potential Benefits of 10% Churn Reduction:**
        - **Monthly revenue saved:** ${:,.2f}
        - **Annual revenue saved:** ${:,.2f}
        - **Customers retained:** {}
        """.format(monthly_revenue_loss * 0.1, annual_revenue_loss * 0.1, int(churn_count * 0.1)))
    else:
        st.info("Monthly charges data not available for revenue analysis")

# ------------------- TAB 5: TIME-TO-CHURN -------------------
with tab5:
    st.subheader("Customer Lifetime Analysis")
    
    st.info("""
    **Time-to-Churn Insights**
    
    Based on typical patterns:
    - Most churn occurs in the first 6 months
    - Customers with longer tenure have lower churn rates
    - Contract type significantly impacts customer lifetime
    """)
    
    # Simulated survival curve (since we can't use Kaplan-Meier without the heavy import)
    months = list(range(0, 73, 6))
    survival_rate = [1.00, 0.85, 0.70, 0.60, 0.55, 0.52, 0.50, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44]
    
    fig = px.line(x=months, y=survival_rate, 
                 title="Typical Customer Survival Curve",
                 labels={'x': 'Months', 'y': 'Survival Probability'})
    fig.add_scatter(x=months, y=survival_rate, mode='markers')
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("""
    **Key Observations:**
    - 15% churn in first 6 months
    - 50% churn by 36 months
    - Long-term customers have stable retention
    """)

# ------------------- TAB 6: NETWORK ANALYSIS -------------------
with tab6:
    st.subheader("🤝 Customer Network Analysis")
    
    # SYNTHETIC DATA GENERATION - Only run when button clicked
    if st.button("🔄 Generate Synthetic Connection Data", key="gen_data_btn"):
        try:
            with st.spinner("Generating connection data..."):
                # Create customer IDs if not exists
                if 'customer_id' not in df.columns:
                    df['customer_id'] = range(len(df))
                
                customer_ids = df['customer_id'].unique()
                
                # Random connections banate hain (SMALLER for demo)
                np.random.seed(42)
                connections = []
                for cust_id in customer_ids[:100]:  # ONLY FIRST 100 CUSTOMERS
                    num_connections = np.random.randint(1, 4)  # Max 3 connections
                    friends = np.random.choice(customer_ids, size=num_connections, replace=False)
                    for friend in friends:
                        if cust_id != friend:
                            connections.append({
                                'customer_id': cust_id, 
                                'connected_id': friend, 
                                'weight': np.random.random(),
                                'connection_type': np.random.choice(['Family', 'Friend', 'Colleague', 'Referral'])
                            })

                connections_df = pd.DataFrame(connections)
                connections_df.to_csv('customer_connections.csv', index=False)
                
            st.success("✅ Synthetic graph data generated!")
            st.dataframe(connections_df.head(10))
            
        except Exception as e:
            st.error(f"Error generating data: {e}")
    
    # EXISTING DATA DISPLAY
    try:
        if os.path.exists('customer_connections.csv'):
            existing_connections = pd.read_csv("customer_connections.csv")
            st.write("### Existing Connections Data")
            st.dataframe(existing_connections.head(5))
            
            col1, col2 = st.columns(2)
            col1.metric("Total Connections", len(existing_connections))
            col2.metric("Unique Customers", existing_connections['customer_id'].nunique())
            
            # ================== NETWORK VISUALIZATION ==================
            st.markdown("---")
            st.subheader("📊 Network Visualization")
            
            # Option 1: Simple NetworkX Visualization
            st.write("### Simple Network Graph")
            try:
                import networkx as nx
                import matplotlib.pyplot as plt
                
                # Create graph
                G = nx.Graph()
                
                # Add nodes and edges
                for _, row in existing_connections.iterrows():
                    G.add_node(row['customer_id'])
                    G.add_node(row['connected_id'])
                    G.add_edge(row['customer_id'], row['connected_id'], 
                              weight=row['weight'],
                              type=row.get('connection_type', 'Unknown'))
                
                # Plot network
                fig, ax = plt.subplots(figsize=(10, 8))
                pos = nx.spring_layout(G, k=0.5, iterations=50)
                
                # Draw nodes and edges
                nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightblue', alpha=0.7)
                nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
                nx.draw_networkx_labels(G, pos, font_size=8)
                
                ax.set_title("Customer Connection Network")
                ax.axis('off')
                st.pyplot(fig)
                
            except ImportError:
                st.warning("NetworkX not installed. Run: pip install networkx matplotlib")
            
            # Option 2: Interactive Pyvis Visualization
            st.write("### Interactive Network Graph")
            try:
                from pyvis.network import Network
                import tempfile
                
                # Create interactive network
                net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
                
                # Sample data for better visualization (first 20 connections)
                sample_connections = existing_connections.head(20)
                
                # Add nodes
                all_nodes = set(sample_connections['customer_id']).union(set(sample_connections['connected_id']))
                for node in all_nodes:
                    net.add_node(node, label=f"Customer {node}", title=f"Customer ID: {node}")
                
                # Add edges
                for _, row in sample_connections.iterrows():
                    net.add_edge(row['customer_id'], row['connected_id'], 
                                title=f"Connection Type: {row.get('connection_type', 'Unknown')}",
                                value=row['weight']*5)
                
                # Generate and show network
                net.save_graph("network.html")
                html_file = open("network.html", 'r', encoding='utf-8')
                html_content = html_file.read()
                html_file.close()
                
                st.components.v1.html(html_content, width=800, height=600)
                
                st.download_button(
                    label="📥 Download Network Graph",
                    data=html_content,
                    file_name="customer_network.html",
                    mime="text/html"
                )
                
            except ImportError:
                st.warning("Pyvis not installed. Run: pip install pyvis")
            
            # Option 3: Network Analysis Metrics
            st.write("### 📈 Network Analysis")
            try:
                import networkx as nx
                
                G = nx.Graph()
                for _, row in existing_connections.iterrows():
                    G.add_edge(row['customer_id'], row['connected_id'])
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Nodes", G.number_of_nodes())
                col2.metric("Total Edges", G.number_of_edges())
                col3.metric("Avg Degree", f"{2*G.number_of_edges()/G.number_of_nodes():.1f}")
                
                # Degree distribution
                if G.number_of_nodes() > 0:
                    degrees = [deg for _, deg in G.degree()]
                    fig = px.histogram(x=degrees, 
                                     title="Degree Distribution - Connections per Customer",
                                     labels={'x': 'Number of Connections', 'y': 'Count'})
                    st.plotly_chart(fig)
                
            except Exception as e:
                st.warning(f"Network analysis skipped: {e}")
            
        else:
            st.info("👆 Click the button above to generate synthetic connection data first.")
            
    except Exception as e:
        st.error(f"Error loading connections: {e}")