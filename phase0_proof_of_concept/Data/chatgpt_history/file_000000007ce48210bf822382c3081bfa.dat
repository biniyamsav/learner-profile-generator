import streamlit as st 
import plotly.express as px
import db
import pandas as pd


def home_page():
    st.header("E-commerce inventory and performance dashboard")
    Analysis=st.button("Analysis")
    if Analysis:
            st.session_state.page = "Analysis"
            st.rerun()
    Management=st.button("Management")
    if Management:
        st.session_state.page = "Management"
        st.rerun()
    Predictive=st.button("Predictive")
    if Predictive:
        st.session_state.page = "Predictive"
        st.rerun()

# def _to_float(value):
#     try:
#         return float(value)
#     except (TypeError, ValueError):
#         return 0.0


# def _first_value(result, index=0, default=None):
#     if result and result[0]:
#         return result[0][index]
#     return default


# def overview_page():
#     st.title("Overview")

#     # ======================
#     # Load Data
#     # ======================
#     revenue_data = db.revenue_per_month_last_12_months()

#     revenue_df = pd.DataFrame(
#         revenue_data,
#         columns=["month", "revenue"]
#     )

#     if revenue_df.empty:
#         st.warning("No revenue data available for the last 12 months.")
#         return

#     revenue_df["revenue"] = revenue_df["revenue"].astype(float)

#     avg_order_value = _to_float(
#         _first_value(db.average_order_value(), default=0)
#     )

#     low_stock_count = len(db.low_stock_products())

#     out_of_stock_count = len(db.out_of_stock_products())

#     top_category = _first_value(
#         db.top_revenue_category(),
#         default="N/A"
#     )

#     worst_supplier = db.worst_on_time_delivery_rate_per_supplier()

#     worst_supplier_name = _first_value(
#         worst_supplier,
#         default="N/A"
#     )

#     worst_supplier_rate = _to_float(
#         _first_value(worst_supplier, index=1, default=0)
#     )

#     new_customers = int(
#         _to_float(
#             _first_value(
#                 db.new_customers_last_month(),
#                 default=0
#             )
#         )
#     )

#     # ======================
#     # Calculate Metrics
#     # ======================
#     revenue_total = revenue_df["revenue"].sum()

#     first_month = revenue_df.iloc[0]["revenue"]
#     last_month = revenue_df.iloc[-1]["revenue"]

#     revenue_delta = (
#         ((last_month - first_month) / first_month) * 100
#         if first_month
#         else 0
#     )

#     # ======================
#     # KPI Cards
#     # ======================
#     col1, col2, col3 = st.columns(3)

#     col1.metric(
#         "Revenue (12 months)",
#         f"${revenue_total:,.2f}",
#         f"{revenue_delta:+.1f}%"
#     )

#     col2.metric(
#         "Average Order Value",
#         f"${avg_order_value:,.2f}"
#     )

#     col3.metric(
#         "New Customers",
#         new_customers
#     )

#     col1, col2, col3 = st.columns(3)

#     col1.metric(
#         "Low Stock Products",
#         low_stock_count
#     )

#     col2.metric(
#         "Out of Stock Products",
#         out_of_stock_count
#     )

#     col3.metric(
#         "Worst Supplier",
#         worst_supplier_name,
#         f"{worst_supplier_rate:.2f}%"
#     )

#     st.markdown(f"**Top Revenue Category:** {top_category}")

#     fig = px.line(
#         revenue_df,
#         x="month",
#         y="revenue",
#         title="Revenue per Month (Last 12 Months)",
#         markers=True
#     )
#     fig.update_layout(xaxis_title="Month", yaxis_title="Revenue ($)")
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown("---")


# def sales_performance_page():
#     st.title("Sales Performance")

#     # =====================================================
#     # Load Data
#     # =====================================================

#     revenue_df = pd.DataFrame(
#         db.revenue_per_month_last_12_months(),
#         columns=["month", "revenue"]
#     )
#     revenue_df["revenue"] = revenue_df["revenue"].astype(float)

#     top_products_df = pd.DataFrame(
#         db.top_10_products_by_revenue(),
#         columns=["product", "revenue"]
#     )
#     top_products_df["revenue"] = top_products_df["revenue"].astype(float)

#     cancelled_df = pd.DataFrame(
#         db.pct_cancelled_vs_delivered(),
#         columns=["status", "percentage"]
#     )
#     cancelled_df["percentage"] = cancelled_df["percentage"].astype(float)

#     rolling_df = pd.DataFrame(
#         db.rolling_7_30_day_sales_per_product(),
#         columns=[
#             "product",
#             "order_date",
#             "daily_sales",
#             "rolling_7_day",
#             "rolling_30_day",
#         ],
#     )

#     if not rolling_df.empty:
#         rolling_df["order_date"] = pd.to_datetime(rolling_df["order_date"])
#         rolling_df["daily_sales"] = rolling_df["daily_sales"].astype(float)
#         rolling_df["rolling_7_day"] = rolling_df["rolling_7_day"].astype(float)
#         rolling_df["rolling_30_day"] = rolling_df["rolling_30_day"].astype(float)

#     monthly_df = pd.DataFrame(
#         db.top_selling_product_per_month_last_12_months(),
#         columns=["product", "month", "quantity_sold"]
#     )
#     monthly_df["quantity_sold"] = monthly_df["quantity_sold"].astype(float)

#     weekday_df = pd.DataFrame(
#         db.highest_order_volume_day_of_week(),
#         columns=["day", "order_volume"]
#     )
#     weekday_df["order_volume"] = weekday_df["order_volume"].astype(int)

#     # =====================================================
#     # KPI Metrics
#     # =====================================================

#     avg_order = _to_float(
#         _first_value(db.average_order_value(), default=0)
#     )

#     category = db.top_revenue_category()
#     category_name = _first_value(category, default="N/A")
#     category_revenue = _to_float(_first_value(category, 1, 0))

#     region = db.top_revenue_region()
#     region_name = _first_value(region, default="N/A")
#     region_revenue = _to_float(_first_value(region, 1, 0))

#     quarter = db.highest_revenue_quarter()

#     quarter_year = _first_value(quarter, default="N/A")
#     quarter_num = _first_value(quarter, 1, "N/A")
#     quarter_revenue = _to_float(_first_value(quarter, 2, 0))

#     c1, c2, c3, c4 = st.columns(4)

#     c1.metric(
#         "Average Order",
#         f"${avg_order:,.2f}"
#     )

#     c2.metric(
#         "Top Category",
#         category_name,
#         f"${category_revenue:,.0f}"
#     )

#     c3.metric(
#         "Top Region",
#         region_name,
#         f"${region_revenue:,.0f}"
#     )

#     c4.metric(
#         "Best Quarter",
#         f"Q{quarter_num}",
#         f"${quarter_revenue:,.0f}"
#     )

#     st.divider()

#     # =====================================================
#     # Revenue Trend
#     # =====================================================

#     revenue_fig = px.line(
#         revenue_df,
#         x="month",
#         y="revenue",
#         markers=True,
#         title="Revenue Trend"
#     )

#     revenue_fig.update_layout(
#         height=450,
#         xaxis_title="Month",
#         yaxis_title="Revenue ($)"
#     )

#     st.plotly_chart(
#         revenue_fig,
#         use_container_width=True
#     )

#     st.divider()

#     # =====================================================
#     # Products + Order Status
#     # =====================================================

#     left, right = st.columns([2, 1])

#     with left:

#         fig = px.bar(
#             top_products_df.sort_values("revenue"),
#             x="revenue",
#             y="product",
#             orientation="h",
#             title="Top 10 Products by Revenue",
#             text_auto=".2s"
#         )

#         fig.update_layout(height=450)

#         st.plotly_chart(
#             fig,
#             use_container_width=True
#         )

#     with right:

#         pie = px.pie(
#             cancelled_df,
#             names="status",
#             values="percentage",
#             hole=.55,
#             title="Order Status"
#         )

#         pie.update_layout(height=450)

#         st.plotly_chart(
#             pie,
#             use_container_width=True
#         )

#     st.divider()

#     # =====================================================
#     # Monthly Winners + Weekday Orders
#     # =====================================================

#     left, right = st.columns(2)

#     with left:

#         monthly_fig = px.bar(
#             monthly_df,
#             x="month",
#             y="quantity_sold",
#             color="product",
#             title="Top Product Per Month"
#         )

#         st.plotly_chart(
#             monthly_fig,
#             use_container_width=True
#         )

#     with right:

#         weekday_fig = px.bar(
#             weekday_df,
#             x="day",
#             y="order_volume",
#             text="order_volume",
#             title="Orders by Day"
#         )

#         st.plotly_chart(
#             weekday_fig,
#             use_container_width=True
#         )

#     st.divider()

#     # =====================================================
#     # Rolling Sales
#     # =====================================================

#     if not rolling_df.empty:

#         st.subheader("Rolling Sales")

#         selected = st.selectbox(
#             "Select Product",
#             sorted(rolling_df["product"].unique())
#         )

#         filtered = rolling_df[
#             rolling_df["product"] == selected
#         ]

#         fig = px.line(
#             filtered,
#             x="order_date",
#             y=["rolling_7_day", "rolling_30_day"],
#             title=f"Rolling Sales - {selected}"
#         )

#         fig.update_layout(height=450)

#         st.plotly_chart(
#             fig,
#             use_container_width=True
#         )

#     st.divider()

#     # =====================================================
#     # Detailed Data
#     # =====================================================

#     with st.expander("Top Products Data"):
#         st.dataframe(
#             top_products_df,
#             use_container_width=True
#         )

#     with st.expander("Monthly Sales Data"):
#         st.dataframe(
#             monthly_df,
#             use_container_width=True
#         )

#     with st.expander("Rolling Sales Data"):
#         st.dataframe(
#             rolling_df,
#             use_container_width=True
#         )

def Analysis_page():
    page = st.sidebar.selectbox(
    "Select Page",
    ["overview", "Sales Performance", "Inventory Health", "Warehouse Performance","Supplier Performance","Customer Analytics","Product Analytics"]
    )

    if page == "overview":
        overview_page()
        

    elif page == "Sales Performance":
        sales_performance_page()
        

    elif page == "Inventory Health":
        st.title("Inventory Health")

    elif page == "Warehouse Performance":
        st.title("Warehouse Performance")
    elif page == "Supplier Performance":
        st.title("Supplier Performance")
    elif page == "Customer Analytics":
        st.title("Customer Analytics")
    elif page == "Product Analytics":
        st.title("Product Analytics")

    Home=st.button("Home")
    if Home:
        st.session_state.page = "Home"
        st.rerun()

def management_page():
    st.header("Management Page")
    Home=st.button("Home")
    if Home:
        st.session_state.page = "Home"
        st.rerun()

def Predictive_page():
    st.header("Predictive page")
    Home=st.button("Home")
    if Home:
        st.session_state.page = "Home"
        st.rerun()

if "page" not in st.session_state:
    st.session_state.page = "Home"
if st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Management":
    management_page()
elif st.session_state.page=="Analysis":
    Analysis_page()
elif st.session_state.page=="Predictive":
    Predictive_page()


