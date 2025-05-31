import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Set page configuration
st.set_page_config(page_title="Customer Segmentation with K-means", layout="wide")

# Set page title
st.title("🛍️ Customer Segmentation with K-means Clustering")
st.markdown("*Understanding your customers through machine learning*")

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = np.array([])
if 'k' not in st.session_state:
    st.session_state.k = 3
if 'max_iterations' not in st.session_state:
    st.session_state.max_iterations = 10
if 'current_iteration' not in st.session_state:
    st.session_state.current_iteration = 0
if 'centroids' not in st.session_state:
    st.session_state.centroids = np.array([])
if 'labels' not in st.session_state:
    st.session_state.labels = np.array([])
if 'inertia_history' not in st.session_state:
    st.session_state.inertia_history = []
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame({'Income': [], 'Spending': []})
if 'previous_centroids' not in st.session_state:
    st.session_state.previous_centroids = np.array([])
if 'show_assignments' not in st.session_state:
    st.session_state.show_assignments = False

# Customer segment names and colours
SEGMENT_NAMES = {
    0: "Budget Shoppers",
    1: "Average Customers", 
    2: "Premium Customers",
    3: "High Earners",
    4: "Luxury Buyers",
    5: "Segment 6",
    6: "Segment 7",
    7: "Segment 8",
    8: "Segment 9",
    9: "Segment 10"
}

SEGMENT_COLOURS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', 
                   '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#576574']

def generate_realistic_customer_data():
    """Generate realistic customer data with income and spending patterns"""
    np.random.seed(42)  # For reproducible results
    
    # Define customer archetypes
    archetypes = [
        {"income_mean": 25000, "income_std": 5000, "spending_mean": 15000, "spending_std": 3000, "size": 25},  # Low income, conservative
        {"income_mean": 45000, "income_std": 8000, "spending_mean": 25000, "spending_std": 5000, "size": 30},  # Middle income, moderate
        {"income_mean": 70000, "income_std": 10000, "spending_mean": 45000, "spending_std": 8000, "size": 20}, # High income, high spending
        {"income_mean": 80000, "income_std": 12000, "spending_mean": 30000, "spending_std": 6000, "size": 15}, # High income, conservative
        {"income_mean": 35000, "income_std": 7000, "spending_mean": 32000, "spending_std": 4000, "size": 10}   # Average income, high spending
    ]
    
    data = []
    for archetype in archetypes:
        for _ in range(archetype["size"]):
            # Generate income (ensuring minimum of £15k)
            income = max(15000, np.random.normal(archetype["income_mean"], archetype["income_std"]))
            
            # Generate spending (ensuring it doesn't exceed income and has minimum of £5k)
            spending_base = np.random.normal(archetype["spending_mean"], archetype["spending_std"])
            spending = max(5000, min(spending_base, income * 0.9))  # Don't spend more than 90% of income
            
            data.append([income, spending])
    
    # Add some noise and outliers
    for _ in range(5):
        income = np.random.uniform(20000, 100000)
        spending = np.random.uniform(5000, min(60000, income * 0.8))
        data.append([income, spending])
    
    st.session_state.data = np.array(data)
    st.session_state.df = pd.DataFrame(data, columns=['Income', 'Spending'])
    st.session_state.centroids = np.array([])
    st.session_state.previous_centroids = np.array([])
    st.session_state.labels = np.array([])
    st.session_state.current_iteration = 0
    st.session_state.inertia_history = []

def initialize_centroids():
    """Initialize centroids randomly within data bounds"""
    if len(st.session_state.data) == 0:
        return
    
    # Store current centroids as previous for movement visualization
    if len(st.session_state.centroids) > 0:
        st.session_state.previous_centroids = st.session_state.centroids.copy()
    
    # Initialize centroids randomly within data bounds
    min_vals = np.min(st.session_state.data, axis=0)
    max_vals = np.max(st.session_state.data, axis=0)
    
    centroids = []
    for _ in range(st.session_state.k):
        centroid = np.random.uniform(min_vals, max_vals)
        centroids.append(centroid)
    
    st.session_state.centroids = np.array(centroids)
    
    # Assign initial labels
    assign_points_to_centroids()
    
    st.session_state.current_iteration = 0
    calculate_inertia()

def assign_points_to_centroids():
    """Assign each point to the nearest centroid"""
    if len(st.session_state.data) == 0 or len(st.session_state.centroids) == 0:
        return
    
    distances = np.sqrt(((st.session_state.data - st.session_state.centroids[:, np.newaxis])**2).sum(axis=2))
    st.session_state.labels = np.argmin(distances, axis=0)

def update_centroids():
    """Update centroids to the mean of assigned points"""
    if len(st.session_state.data) == 0 or len(st.session_state.labels) == 0:
        return False
    
    # Store previous centroids for movement visualization
    st.session_state.previous_centroids = st.session_state.centroids.copy()
    
    new_centroids = []
    centroid_moved = False
    
    for i in range(st.session_state.k):
        cluster_points = st.session_state.data[st.session_state.labels == i]
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
        else:
            # If no points assigned, keep current centroid
            new_centroid = st.session_state.centroids[i]
        
        # Check if centroid moved significantly
        if len(st.session_state.centroids) > 0:
            movement = np.sqrt(np.sum((new_centroid - st.session_state.centroids[i])**2))
            if movement > 500:  # Threshold for "significant" movement in pounds
                centroid_moved = True
        
        new_centroids.append(new_centroid)
    
    st.session_state.centroids = np.array(new_centroids)
    return centroid_moved

def run_iteration():
    """Perform one complete iteration of k-means"""
    if len(st.session_state.data) == 0 or len(st.session_state.centroids) == 0:
        return False
    
    # Step 1: Assign points to nearest centroids
    assign_points_to_centroids()
    
    # Step 2: Update centroids
    centroids_moved = update_centroids()
    
    # Step 3: Reassign points to updated centroids
    assign_points_to_centroids()
    
    st.session_state.current_iteration += 1
    calculate_inertia()
    
    return centroids_moved

def calculate_inertia():
    """Calculate and store current inertia (within-cluster sum of squares)"""
    if len(st.session_state.data) == 0 or len(st.session_state.centroids) == 0:
        return
    
    inertia = 0
    for i in range(st.session_state.k):
        cluster_points = st.session_state.data[st.session_state.labels == i]
        if len(cluster_points) > 0:
            centroid = st.session_state.centroids[i]
            inertia += np.sum((cluster_points - centroid)**2)
    
    # Update or append to history
    current_entry = {"iteration": st.session_state.current_iteration, "inertia": inertia}
    
    if st.session_state.current_iteration == 0:
        st.session_state.inertia_history = [current_entry]
    else:
        # Update existing entry or append new one
        if len(st.session_state.inertia_history) > st.session_state.current_iteration:
            st.session_state.inertia_history[st.session_state.current_iteration] = current_entry
        else:
            st.session_state.inertia_history.append(current_entry)

def draw_customer_visualization():
    """Create the main clustering visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if len(st.session_state.data) > 0:
        # Plot customer data points
        if len(st.session_state.labels) > 0:
            for i in range(st.session_state.k):
                cluster_points = st.session_state.data[st.session_state.labels == i]
                if len(cluster_points) > 0:
                    ax.scatter(
                        cluster_points[:, 0] / 1000,  # Convert to thousands for readability
                        cluster_points[:, 1] / 1000,
                        c=SEGMENT_COLOURS[i % len(SEGMENT_COLOURS)],
                        alpha=0.7,
                        s=60,
                        label=SEGMENT_NAMES.get(i, f'Segment {i+1}'),
                        edgecolors='white',
                        linewidth=0.5
                    )
        else:
            ax.scatter(
                st.session_state.data[:, 0] / 1000,
                st.session_state.data[:, 1] / 1000,
                c='#999999',
                alpha=0.7,
                s=60,
                edgecolors='white',
                linewidth=0.5
            )
        
        # Plot centroids
        if len(st.session_state.centroids) > 0:
            ax.scatter(
                st.session_state.centroids[:, 0] / 1000,
                st.session_state.centroids[:, 1] / 1000,
                c='black',
                s=200,
                marker='X',
                label='Centroids',
                edgecolors='white',
                linewidth=2
            )
            
            # Show centroid movement if available
            if len(st.session_state.previous_centroids) > 0 and st.session_state.current_iteration > 0:
                for i in range(len(st.session_state.centroids)):
                    if i < len(st.session_state.previous_centroids):
                        ax.annotate('', 
                                  xy=(st.session_state.centroids[i, 0] / 1000, st.session_state.centroids[i, 1] / 1000),
                                  xytext=(st.session_state.previous_centroids[i, 0] / 1000, st.session_state.previous_centroids[i, 1] / 1000),
                                  arrowprops=dict(arrowstyle='->', lw=2, color='red', alpha=0.7))
        
        # Show assignment lines if requested
        if st.session_state.show_assignments and len(st.session_state.centroids) > 0 and len(st.session_state.labels) > 0:
            for i, point in enumerate(st.session_state.data):
                centroid = st.session_state.centroids[st.session_state.labels[i]]
                ax.plot([point[0]/1000, centroid[0]/1000], 
                       [point[1]/1000, centroid[1]/1000], 
                       'k-', alpha=0.2, linewidth=0.5)
    
    # Formatting
    ax.set_xlabel('Annual Income (£ thousands)', fontsize=12)
    ax.set_ylabel('Annual Spending (£ thousands)', fontsize=12)
    ax.set_title('Customer Segmentation Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if len(st.session_state.data) > 0:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set reasonable axis limits
    if len(st.session_state.data) > 0:
        x_margin = (np.max(st.session_state.data[:, 0]) - np.min(st.session_state.data[:, 0])) * 0.1
        y_margin = (np.max(st.session_state.data[:, 1]) - np.min(st.session_state.data[:, 1])) * 0.1
        ax.set_xlim((np.min(st.session_state.data[:, 0]) - x_margin) / 1000, 
                   (np.max(st.session_state.data[:, 0]) + x_margin) / 1000)
        ax.set_ylim((np.min(st.session_state.data[:, 1]) - y_margin) / 1000, 
                   (np.max(st.session_state.data[:, 1]) + y_margin) / 1000)
    
    plt.tight_layout()
    return fig

def calculate_elbow_curve():
    """Calculate inertia for different numbers of clusters"""
    if len(st.session_state.data) == 0:
        return None, None
    
    k_range = range(1, min(11, len(st.session_state.data)))
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(st.session_state.data)
        inertias.append(kmeans.inertia_)
    
    return list(k_range), inertias

def draw_elbow_plot():
    """Create elbow method visualization"""
    k_values, inertias = calculate_elbow_curve()
    
    if k_values is None:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    
    # Highlight current k value
    if st.session_state.k in k_values:
        current_idx = k_values.index(st.session_state.k)
        ax.plot(st.session_state.k, inertias[current_idx], 'ro', markersize=12, label=f'Current k={st.session_state.k}')
    
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    ax.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

# Main app layout
st.write("## 📊 Data Generation")

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🎲 Generate Customer Data", type="primary"):
        generate_realistic_customer_data()
        st.success("Realistic customer data generated!")

with col2:
    if len(st.session_state.data) > 0:
        st.metric("Total Customers", len(st.session_state.data))

# Display sample data
if len(st.session_state.data) > 0:
    st.write("### Sample Customer Data")
    display_df = st.session_state.df.copy()
    display_df['Income'] = display_df['Income'].apply(lambda x: f"£{x:,.0f}")
    display_df['Spending'] = display_df['Spending'].apply(lambda x: f"£{x:,.0f}")
    st.dataframe(display_df.head(10), use_container_width=True)

st.markdown("---")

# Main visualization and controls
col1, col2 = st.columns([3, 1])

with col1:
    st.write("## 🎯 Customer Segmentation")
    
    if len(st.session_state.data) > 0:
        fig = draw_customer_visualization()
        st.pyplot(fig)
        
        # Visualization options
        st.session_state.show_assignments = st.checkbox(
            "Show point-to-centroid assignments", 
            value=st.session_state.show_assignments,
            help="Display lines connecting each customer to their assigned segment centroid"
        )
    else:
        st.info("👆 Generate customer data first to see the segmentation analysis.")

with col2:
    st.write("## ⚙️ Algorithm Controls")
    
    if len(st.session_state.data) > 0:
        # Number of clusters
        new_k = st.slider(
            "Number of segments (k)", 
            min_value=1, 
            max_value=min(10, len(st.session_state.data)), 
            value=st.session_state.k,
            help="Choose how many customer segments to create"
        )
        
        if new_k != st.session_state.k:
            st.session_state.k = new_k
            # Reset algorithm state when k changes
            st.session_state.centroids = np.array([])
            st.session_state.current_iteration = 0
            st.session_state.inertia_history = []
        
        # Initialize button
        if st.button("🎯 Initialize Centroids", type="primary"):
            initialize_centroids()
            st.rerun()
        
        # Step-by-step execution
        step_disabled = (len(st.session_state.centroids) == 0 or 
                        st.session_state.current_iteration >= st.session_state.max_iterations)
        
        if st.button("▶️ Run One Iteration", disabled=step_disabled):
            changed = run_iteration()
            if not changed:
                st.success("✅ Algorithm converged!")
            st.rerun()
        
        # Run to completion
        if st.button("⏩ Run Until Convergence") and not step_disabled:
            max_iter = 20
            for _ in range(max_iter):
                if not run_iteration():
                    break
            st.success("🎉 Algorithm completed!")
            st.rerun()
        
        # Algorithm status
        st.write("### 📈 Status")
        st.write(f"**Iteration:** {st.session_state.current_iteration}")
        
        if len(st.session_state.inertia_history) > 0:
            current_inertia = st.session_state.inertia_history[-1]["inertia"]
            st.write(f"**Inertia:** {current_inertia:,.0f}")
            st.caption("Lower inertia = better clustering")

# Algorithm progress visualization
if len(st.session_state.inertia_history) > 1:
    st.write("## 📉 Algorithm Progress")
    
    col1, col2 = st.columns(2)
    
    with col1:
        history_df = pd.DataFrame(st.session_state.inertia_history)
        st.line_chart(history_df.set_index("iteration")["inertia"], use_container_width=True)
        st.caption("Inertia decreases as the algorithm finds better cluster centres")
    
    with col2:
        if len(st.session_state.inertia_history) >= 2:
            improvements = []
            for i in range(1, len(st.session_state.inertia_history)):
                prev_inertia = st.session_state.inertia_history[i-1]["inertia"]
                curr_inertia = st.session_state.inertia_history[i]["inertia"]
                improvement = ((prev_inertia - curr_inertia) / prev_inertia) * 100
                improvements.append({"iteration": i, "improvement": improvement})
            
            if improvements:
                improvement_df = pd.DataFrame(improvements)
                st.bar_chart(improvement_df.set_index("iteration")["improvement"], use_container_width=True)
                st.caption("Percentage improvement in each iteration")

# Elbow method for optimal k
st.write("## 🔍 Finding the Optimal Number of Clusters")

if len(st.session_state.data) > 0:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        elbow_fig = draw_elbow_plot()
        if elbow_fig:
            st.pyplot(elbow_fig)
    
    with col2:
        st.write("### 💡 How to Use the Elbow Method")
        st.markdown("""
        1. **Look for the 'elbow'** - the point where the inertia starts decreasing more slowly
        2. **Balance complexity vs. performance** - more clusters isn't always better
        3. **Consider business needs** - sometimes you want a specific number of segments
        """)
        
        if st.button("📊 Calculate Elbow Analysis"):
            k_values, inertias = calculate_elbow_curve()
            if k_values and len(k_values) > 2:
                # Simple elbow detection
                improvements = []
                for i in range(1, len(inertias)):
                    improvement = inertias[i-1] - inertias[i]
                    improvements.append(improvement)
                
                # Find where improvement starts to level off
                if len(improvements) > 1:
                    rates = []
                    for i in range(1, len(improvements)):
                        rate = improvements[i-1] - improvements[i]
                        rates.append(rate)
                    
                    if rates:
                        suggested_k = rates.index(max(rates)) + 2  # +2 because we start from k=1 and index adjustment
                        st.info(f"💡 Suggested k: {suggested_k}")

# Educational content
st.markdown("---")
st.write("## 🎓 Understanding Customer Segmentation")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🔄 How K-means Works:
    1. **Choose k** (number of segments)
    2. **Place random centroids** in the data space
    3. **Assign customers** to nearest centroid
    4. **Move centroids** to the centre of their assigned customers
    5. **Repeat steps 3-4** until centroids stop moving
    """)

with col2:
    st.markdown("""
    ### 💼 Business Applications:
    - **Marketing campaigns** - target specific customer groups
    - **Product development** - design for different segments
    - **Pricing strategies** - optimise for each segment
    - **Inventory management** - stock based on segment preferences
    - **Customer service** - tailor support approaches
    """)

# Segment analysis
if len(st.session_state.data) > 0 and len(st.session_state.labels) > 0:
    st.write("## 👥 Segment Analysis")
    
    segment_stats = []
    for i in range(st.session_state.k):
        cluster_data = st.session_state.data[st.session_state.labels == i]
        if len(cluster_data) > 0:
            segment_stats.append({
                "Segment": SEGMENT_NAMES.get(i, f"Segment {i+1}"),
                "Customers": len(cluster_data),
                "Avg Income": f"£{np.mean(cluster_data[:, 0]):,.0f}",
                "Avg Spending": f"£{np.mean(cluster_data[:, 1]):,.0f}",
                "Spending Ratio": f"{np.mean(cluster_data[:, 1]) / np.mean(cluster_data[:, 0]) * 100:.1f}%"
            })
    
    if segment_stats:
        st.dataframe(pd.DataFrame(segment_stats), use_container_width=True)
        st.caption("💡 Spending Ratio = Average Spending ÷ Average Income × 100%")