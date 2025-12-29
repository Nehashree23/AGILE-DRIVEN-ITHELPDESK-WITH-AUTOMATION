
import streamlit as st
import torch
import numpy as np
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datetime import datetime
import sqlite3
import plotly.express as px

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="IT Helpdesk Dashboard", page_icon="🛠️", layout="wide")

st.title("🛠️ IT Helpdesk System")
st.write("Employee & Admin Interface for ticket submission, routing, and prioritization.")

# -------------------------------
# Load Model & Tokenizer
# -------------------------------
@st.cache_resource
def load_model():
    model_path = "/content/drive/MyDrive/it_helpdesk_roberta_model"
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

# -------------------------------
# Label Mapping & Routing
# -------------------------------
label_map = {
    0: "Access Issues",
    1: "Email Issues",
    2: "Hardware Issues",
    3: "Network Issues",
    4: "Security Issues",
    5: "Software Issues",
    6: "Account Issues",
    7: "Other Issues"
}

routing_table = {
    "Access Issues": "Access Team",
    "Email Issues": "Email Support",
    "Hardware Issues": "Hardware Team",
    "Network Issues": "Network Team",
    "Security Issues": "Security Team",
    "Software Issues": "Software Support",
    "Account Issues": "Account Management",
    "Other Issues": "Review manually"
}

urgency_colors = {"High": "#ff4d4d", "Medium": "#ffa500", "Low": "#2ecc71"}

# -------------------------------
# Database Setup (SQLite)
# -------------------------------
conn = sqlite3.connect("tickets.db", check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS tickets(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    text TEXT,
    category TEXT,
    confidence REAL,
    team TEXT,
    urgency TEXT,
    status TEXT
)
''')
conn.commit()

# -------------------------------
# Tabs: Employee / Admin
# -------------------------------
tabs = st.tabs(["Employee", "Admin"])

# -------------------------------
# Employee Tab
# -------------------------------
with tabs[0]:
    st.subheader("📩 Submit a Ticket")
    ticket_text = st.text_area("Ticket Description", height=150, placeholder="Describe your issue...")
    urgency = st.radio("Select Urgency Level", ["Low", "Medium", "High"], index=1)
    
    # Live prediction suggestion
    if ticket_text.strip() != "":
        inputs = tokenizer(ticket_text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(probs)
            suggested_category = label_map[pred_class]
            st.info(f"💡 Suggested Category: **{suggested_category}** ({probs[pred_class]*100:.1f}%)")
    
    if st.button("📝 Submit Ticket"):
        if ticket_text.strip() == "":
            st.warning("Enter ticket description.")
        else:
            # Save to SQLite
            c.execute('''
                INSERT INTO tickets (timestamp, text, category, confidence, team, urgency, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (datetime.now(), ticket_text, suggested_category, probs[pred_class],
                  routing_table[suggested_category], urgency, "New"))
            conn.commit()
            st.success(f"Ticket submitted! Routed to {routing_table[suggested_category]} | Urgency: {urgency}")

# -------------------------------
# Admin Tab
# -------------------------------
with tabs[1]:
    st.subheader("🖥️ Admin Dashboard")
    
    # Filters
    df = pd.read_sql("SELECT * FROM tickets", conn)
    if df.empty:
        st.info("No tickets submitted yet.")
    else:
        # Filters
        categories = st.multiselect("Filter by Category", options=df['category'].unique(), default=df['category'].unique())
        teams = st.multiselect("Filter by Team", options=df['team'].unique(), default=df['team'].unique())
        urgency_filter = st.multiselect("Filter by Urgency", options=df['urgency'].unique(), default=df['urgency'].unique())
        status_filter = st.multiselect("Filter by Status", options=df['status'].unique(), default=df['status'].unique())
        
        df_filtered = df[df['category'].isin(categories) & df['team'].isin(teams) &
                         df['urgency'].isin(urgency_filter) & df['status'].isin(status_filter)]
        
        # Sort by urgency High->Low, then FIFO
        urgency_map = {"High": 3, "Medium": 2, "Low": 1}
        df_filtered['urgency_level'] = df_filtered['urgency'].map(urgency_map)
        df_filtered = df_filtered.sort_values(by=['urgency_level', 'timestamp'], ascending=[False, True])
        
        # Editable status column
        for idx, row in df_filtered.iterrows():
            with st.expander(f"📝 Ticket #{row['id']} | {row['urgency']} | {row['category']} | Team: {row['team']}"):
                st.write(f"**Description:** {row['text']}")
                new_status = st.selectbox("Update Status", ["New", "In Progress", "Resolved"], index=["New","In Progress","Resolved"].index(row['status']), key=row['id'])
                if new_status != row['status']:
                    c.execute("UPDATE tickets SET status=? WHERE id=?", (new_status, row['id']))
                    conn.commit()
                    st.success("Status updated!")
        
        # Dashboard Metrics
        st.markdown("### 📊 Dashboard Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tickets", len(df_filtered))
        col2.metric("High Urgency", len(df_filtered[df_filtered['urgency']=="High"]))
        col3.metric("Teams", len(df_filtered['team'].unique()))
        col4.metric("Resolved Tickets", len(df_filtered[df_filtered['status']=="Resolved"]))
        
        # Interactive Charts
        st.markdown("### Tickets per Category")
        fig1 = px.bar(df_filtered, x='category', color='urgency', color_discrete_map=urgency_colors)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("### Tickets per Team")
        fig2 = px.bar(df_filtered, x='team', color='urgency', color_discrete_map=urgency_colors)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("### Category Heatmap over Time (Hour)")
        df_filtered['hour'] = pd.to_datetime(df_filtered['timestamp']).dt.floor('H')
        heatmap = df_filtered.groupby(['hour', 'category']).size().reset_index(name='count')
        fig3 = px.density_heatmap(heatmap, x='hour', y='category', z='count', color_continuous_scale='Viridis')
        st.plotly_chart(fig3, use_container_width=True)
