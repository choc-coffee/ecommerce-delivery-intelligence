from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# Basic app setup
st.set_page_config(
    page_title="E-commerce Delivery Intelligence",
    page_icon="📦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "master_orders.csv"
MODEL_PATH = BASE_DIR / "models" / "late_delivery_model.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "classification_threshold.joblib"


# Load the processed order data
@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
    )


# Load the model and the threshold chosen during validation
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    return model, threshold


# Stop early if any of the files the app needs are missing
if not DATA_PATH.exists():
    st.error(
        "Couldn't find the processed dataset at "
        "`data/processed/master_orders.csv`."
    )
    st.stop()

if not MODEL_PATH.exists():
    st.error(
        "Couldn't find the trained model at "
        "`models/late_delivery_model.joblib`."
    )
    st.stop()

if not THRESHOLD_PATH.exists():
    st.error(
        "Couldn't find the saved classification threshold at "
        "`models/classification_threshold.joblib`."
    )
    st.stop()


df = load_data()
model, threshold = load_model()


# Page heading
st.title("📦 E-commerce Delivery Intelligence")

st.write(
    "An interactive look at delivery performance across the "
    "Olist Brazilian e-commerce dataset."
)

st.caption(
    "The project looks at what is linked with late delivery, "
    "how delays affect customer reviews, and whether higher-risk "
    "orders can be identified earlier."
)


# Main sections of the app
overview_tab, analysis_tab, prediction_tab, recommendations_tab = st.tabs(
    [
        "Overview",
        "Delivery Analysis",
        "Prediction",
        "Recommendations"
    ]
)


# OVERVIEW

with overview_tab:

    st.header("Overview")

    total_orders = len(df)
    total_order_value = df["order_value"].sum()
    average_order_value = df["order_value"].mean()
    late_rate = df["is_late"].mean() * 100
    average_review = df["review_score"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Orders",
        f"{total_orders:,}"
    )

    col2.metric(
        "Total Order Value",
        f"R${total_order_value:,.0f}"
    )

    col3.metric(
        "Average Order Value",
        f"R${average_order_value:,.2f}"
    )

    col4.metric(
        "Late Delivery Rate",
        f"{late_rate:.2f}%"
    )

    col5.metric(
        "Average Review",
        f"{average_review:.2f}/5"
    )

    st.divider()

    st.subheader("Late delivery and customer reviews")

    review_summary = (
        df
        .groupby("is_late")
        .agg(
            average_review=("review_score", "mean")
        )
        .reset_index()
    )

    review_summary["delivery_status"] = (
        review_summary["is_late"]
        .map({
            0: "On time",
            1: "Late"
        })
    )

    fig_review = px.bar(
        review_summary,
        x="delivery_status",
        y="average_review",
        title="Average Review Score by Delivery Status",
        labels={
            "delivery_status": "Delivery Status",
            "average_review": "Average Review Score"
        },
        text_auto=".2f"
    )

    fig_review.update_layout(
        yaxis_range=[0, 5]
    )

    st.plotly_chart(
        fig_review,
        use_container_width=True
    )

    st.info(
        "On-time orders averaged 4.29/5 in customer reviews, "
        "compared with 2.57/5 for late orders."
    )

    st.divider()

    st.subheader("Order volume over time")

    monthly_orders = (
        df
        .groupby("purchase_year_month")
        .agg(
            orders=("order_id", "count"),
            order_value=("order_value", "sum")
        )
        .reset_index()
    )

    fig_orders = px.line(
        monthly_orders,
        x="purchase_year_month",
        y="orders",
        markers=True,
        title="Monthly Order Volume",
        labels={
            "purchase_year_month": "Month",
            "orders": "Orders"
        }
    )

    st.plotly_chart(
        fig_orders,
        use_container_width=True
    )


# DELIVERY ANALYSIS

with analysis_tab:

    st.header("Delivery Analysis")

    st.write(
        "This section looks at where and when late deliveries were "
        "more common, and which order characteristics were linked "
        "with higher delivery risk."
    )

    # Late delivery over time
    st.subheader("Late delivery over time")

    monthly_delivery = (
        df
        .groupby("purchase_year_month")
        .agg(
            orders=("order_id", "count"),
            late_delivery_rate=("is_late", "mean")
        )
        .reset_index()
    )

    monthly_delivery["late_delivery_rate"] *= 100

    fig_monthly = px.line(
        monthly_delivery,
        x="purchase_year_month",
        y="late_delivery_rate",
        markers=True,
        title="Late Delivery Rate Over Time",
        labels={
            "purchase_year_month": "Month",
            "late_delivery_rate": "Late Delivery Rate (%)"
        }
    )

    st.plotly_chart(
        fig_monthly,
        use_container_width=True
    )

    st.caption(
        "The late-delivery rate changes quite a bit over time, "
        "which is one reason the model was tested on later orders "
        "rather than using a random train/test split."
    )

    st.divider()

    # Customer state
    st.subheader("Delivery performance by customer state")

    state_summary = (
        df
        .groupby("customer_state")
        .agg(
            orders=("order_id", "count"),
            late_delivery_rate=("is_late", "mean"),
            average_delivery_days=("actual_delivery_days", "mean")
        )
        .reset_index()
    )

    state_summary["late_delivery_rate"] *= 100

    # Very small states can give misleading percentages,
    # so only show states with at least 100 orders
    state_summary = state_summary[
        state_summary["orders"] >= 100
    ]

    fig_state = px.bar(
        state_summary.sort_values(
            "late_delivery_rate",
            ascending=False
        ),
        x="customer_state",
        y="late_delivery_rate",
        hover_data=[
            "orders",
            "average_delivery_days"
        ],
        title="Late Delivery Rate by Customer State",
        labels={
            "customer_state": "Customer State",
            "late_delivery_rate": "Late Delivery Rate (%)",
            "orders": "Orders",
            "average_delivery_days": "Average Delivery Days"
        }
    )

    st.plotly_chart(
        fig_state,
        use_container_width=True
    )

    st.info(
        "Alagoas (AL) had the highest late-delivery rate among "
        "states with at least 100 orders: 23.93%, compared with "
        "8.11% across the dataset overall."
    )

    st.divider()

    # Product category
    st.subheader("Delivery performance by product category")

    category_summary = (
        df
        .groupby("primary_product_category")
        .agg(
            orders=("order_id", "count"),
            late_delivery_rate=("is_late", "mean")
        )
        .reset_index()
    )

    category_summary["late_delivery_rate"] *= 100

    # Again, keep enough orders in each group for a fairer comparison
    category_summary = category_summary[
        category_summary["orders"] >= 300
    ]

    top_categories = (
        category_summary
        .sort_values(
            "late_delivery_rate",
            ascending=False
        )
        .head(15)
    )

    fig_category = px.bar(
        top_categories,
        x="late_delivery_rate",
        y="primary_product_category",
        orientation="h",
        hover_data=["orders"],
        title="Categories with the Highest Late Delivery Rates",
        labels={
            "late_delivery_rate": "Late Delivery Rate (%)",
            "primary_product_category": "Product Category",
            "orders": "Orders"
        }
    )

    fig_category.update_yaxes(
        categoryorder="total ascending"
    )

    st.plotly_chart(
        fig_category,
        use_container_width=True
    )

    st.caption(
        "Audio had the highest late-delivery rate among categories "
        "with at least 300 orders."
    )

    st.divider()

    # Freight and promised delivery window
    st.subheader("Order characteristics linked with late delivery")

    left_col, right_col = st.columns(2)

    with left_col:

        freight_df = df.copy()

        freight_df["freight_group"] = pd.qcut(
            freight_df["freight_value"],
            q=5,
            labels=[
                "Lowest",
                "Low",
                "Medium",
                "High",
                "Highest"
            ]
        )

        freight_summary = (
            freight_df
            .groupby(
                "freight_group",
                observed=True
            )
            .agg(
                orders=("order_id", "count"),
                average_freight=("freight_value", "mean"),
                late_delivery_rate=("is_late", "mean")
            )
            .reset_index()
        )

        freight_summary["late_delivery_rate"] *= 100

        fig_freight = px.bar(
            freight_summary,
            x="freight_group",
            y="late_delivery_rate",
            title="Late Rate by Freight Cost",
            labels={
                "freight_group": "Freight Cost Group",
                "late_delivery_rate": "Late Delivery Rate (%)"
            },
            hover_data=[
                "orders",
                "average_freight"
            ]
        )

        st.plotly_chart(
            fig_freight,
            use_container_width=True
        )

    with right_col:

        promise_df = df.copy()

        promise_df["promise_window_group"] = pd.qcut(
            promise_df["promised_delivery_days"],
            q=5,
            labels=[
                "Shortest",
                "Short",
                "Medium",
                "Long",
                "Longest"
            ]
        )

        promise_summary = (
            promise_df
            .groupby(
                "promise_window_group",
                observed=True
            )
            .agg(
                orders=("order_id", "count"),
                average_promised_days=(
                    "promised_delivery_days",
                    "mean"
                ),
                late_delivery_rate=("is_late", "mean")
            )
            .reset_index()
        )

        promise_summary["late_delivery_rate"] *= 100

        fig_promise = px.bar(
            promise_summary,
            x="promise_window_group",
            y="late_delivery_rate",
            title="Late Rate by Promised Delivery Window",
            labels={
                "promise_window_group": "Promised Window",
                "late_delivery_rate": "Late Delivery Rate (%)"
            },
            hover_data=[
                "orders",
                "average_promised_days"
            ]
        )

        st.plotly_chart(
            fig_promise,
            use_container_width=True
        )

    st.info(
        "Late deliveries became more common as freight costs increased. "
        "Longer promised delivery windows were generally linked with "
        "lower late-delivery rates. These are relationships in the data, "
        "not proof that one factor directly caused the other."
    )

    st.divider()

    # Review score distribution
    st.subheader("What happens to customer reviews when an order is late?")

    review_distribution = (
        df
        .groupby(
            [
                "is_late",
                "review_score"
            ]
        )
        .size()
        .reset_index(
            name="orders"
        )
    )

    review_distribution["percentage"] = (
        review_distribution
        .groupby("is_late")["orders"]
        .transform(
            lambda x: x / x.sum() * 100
        )
    )

    review_distribution["delivery_status"] = (
        review_distribution["is_late"]
        .map({
            0: "On time",
            1: "Late"
        })
    )

    fig_reviews = px.bar(
        review_distribution,
        x="review_score",
        y="percentage",
        color="delivery_status",
        barmode="group",
        title="Review Score Distribution by Delivery Status",
        labels={
            "review_score": "Review Score",
            "percentage": "Orders (%)",
            "delivery_status": "Delivery Status"
        }
    )

    st.plotly_chart(
        fig_reviews,
        use_container_width=True
    )

    st.info(
        "The difference is pretty clear: late orders averaged "
        "2.57/5, while on-time orders averaged 4.29/5."
    )


# PREDICTION

with prediction_tab:

    st.header("Late-Delivery Risk Prediction")

    st.write(
        "Change the order details below to see how the model ranks "
        "its delivery risk."
    )

    st.warning(
        "The score is useful for comparing relative risk between orders. "
        "It is not the same as saying an order has that exact percentage "
        "chance of arriving late."
    )

    input_col1, input_col2 = st.columns(2)

    with input_col1:

        order_value = st.number_input(
            "Order Value (R$)",
            min_value=0.0,
            value=100.0,
            step=10.0
        )

        freight_value = st.number_input(
            "Freight Value (R$)",
            min_value=0.0,
            value=20.0,
            step=5.0
        )

        number_of_items = st.number_input(
            "Number of Items",
            min_value=1,
            value=1,
            step=1
        )

        customer_state = st.selectbox(
            "Customer State",
            sorted(
                df["customer_state"]
                .dropna()
                .unique()
            )
        )

        primary_payment_type = st.selectbox(
            "Payment Type",
            sorted(
                df["primary_payment_type"]
                .dropna()
                .unique()
            )
        )

    with input_col2:

        number_of_categories = st.number_input(
            "Number of Product Categories",
            min_value=1,
            value=1,
            step=1
        )

        number_of_sellers = st.number_input(
            "Number of Sellers",
            min_value=1,
            value=1,
            step=1
        )

        primary_seller_state = st.selectbox(
            "Primary Seller State",
            sorted(
                df["primary_seller_state"]
                .dropna()
                .unique()
            )
        )

        primary_product_category = st.selectbox(
            "Primary Product Category",
            sorted(
                df["primary_product_category"]
                .dropna()
                .unique()
            )
        )

        max_payment_installments = st.number_input(
            "Payment Instalments",
            min_value=0,
            value=1,
            step=1
        )

    purchase_month = st.selectbox(
        "Purchase Month",
        list(range(1, 13))
    )

    purchase_weekday = st.selectbox(
        "Purchase Weekday",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )

    promised_delivery_days = st.number_input(
        "Promised Delivery Window (Days)",
        min_value=1.0,
        value=20.0,
        step=1.0
    )

    if st.button(
        "Assess Delivery Risk",
        type="primary"
    ):

        input_data = pd.DataFrame(
            [{
                "order_value": order_value,
                "freight_value": freight_value,
                "number_of_items": number_of_items,
                "number_of_categories": number_of_categories,
                "number_of_sellers": number_of_sellers,
                "customer_state": customer_state,
                "primary_seller_state": primary_seller_state,
                "primary_product_category": primary_product_category,
                "primary_payment_type": primary_payment_type,
                "max_payment_installments": max_payment_installments,
                "purchase_month": purchase_month,
                "purchase_weekday": purchase_weekday,
                "promised_delivery_days": promised_delivery_days
            }]
        )

        risk_score = model.predict_proba(
            input_data
        )[0, 1]

        if risk_score < 0.40:
            risk_tier = "Lower Risk"

        elif risk_score < threshold:
            risk_tier = "Moderate Risk"

        else:
            risk_tier = "Higher Risk"

        st.divider()

        st.subheader(
            f"Result: {risk_tier}"
        )

        st.progress(
            float(risk_score)
        )

        st.caption(
            f"Model risk score: {risk_score:.2f}. "
            "This is a relative risk score rather than a calibrated "
            "probability."
        )

        if risk_tier == "Higher Risk":

            st.warning(
                "This order sits above the model's high-risk threshold "
                "and could be worth monitoring more closely."
            )

        elif risk_tier == "Moderate Risk":

            st.info(
                "This order sits in the middle of the model's risk range."
            )

        else:

            st.success(
                "This order falls into the model's lower-risk range."
            )

    st.divider()

    st.subheader("How well did the model perform?")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Test ROC-AUC",
        "0.717"
    )

    metric2.metric(
        "Top Risk Decile Lift",
        "1.99×"
    )

    metric3.metric(
        "Test Late Rate",
        "5.29%"
    )

    st.write(
        "The model was tested on a later period of orders that it had "
        "not seen during training. The highest-risk 10% of orders had "
        "a 10.52% late-delivery rate, compared with 5.29% across the "
        "test set overall."
    )

    st.caption(
        "The model works better as a way to rank and prioritise orders "
        "than as a yes/no prediction of whether an order will be late."
    )


# RECOMMENDATIONS

with recommendations_tab:

    st.header("Recommendations")

    st.write(
        "A few practical takeaways from the analysis."
    )

    st.divider()

    st.subheader("1. Focus attention on higher-risk orders")

    st.write(
        "The model should be used to help decide which orders deserve "
        "closer monitoring, rather than automatically deciding which "
        "orders will be late."
    )

    st.write(
        "The highest-risk 10% of orders were about **1.99x as likely "
        "to arrive late** as an average order in the test period."
    )

    st.divider()

    st.subheader("2. Look more closely at regional delivery problems")

    st.write(
        "**Alagoas (AL) had a 23.93% late-delivery rate**, compared "
        "with **8.11% overall**. Some other states also performed well "
        "above the overall late rate."
    )

    st.write(
        "That makes geography one of the clearest areas to investigate "
        "further, particularly around fulfilment routes and regional "
        "shipping constraints."
    )

    st.divider()

    st.subheader("3. Treat delivery performance as a customer issue too")

    st.write(
        "Late orders averaged **2.57/5** in customer reviews, compared "
        "with **4.29/5** for orders delivered on time."
    )

    st.write(
        "That suggests delivery problems are not just an operations "
        "issue. They also have a clear link with customer experience."
    )

    st.divider()

    st.subheader("4. Pay attention to freight cost and delivery promises")

    st.write(
        "Orders in the highest freight-cost group had a "
        "**9.35% late-delivery rate**, compared with **6.08%** "
        "in the lowest group."
    )

    st.write(
        "Orders with the longest promised delivery windows had a "
        "**5.41% late rate**, compared with roughly **8–10%** across "
        "the shorter groups."
    )

    st.write(
        "These patterns would be worth investigating further alongside "
        "distance, shipping routes and how delivery estimates are set."
    )

    st.divider()

    st.subheader("5. Keep the model in perspective")

    st.write(
        "The final Logistic Regression model reached a **0.717 ROC-AUC** "
        "on later unseen orders."
    )

    st.write(
        "That shows the model can separate higher-risk from lower-risk "
        "orders reasonably well, but it still produces a lot of false "
        "positives. In practice, it makes more sense as a prioritisation "
        "tool than an automated decision system."
    )

    st.divider()

    st.caption(
        "These results show patterns in the historical Olist data. "
        "They do not prove that any one factor directly causes late delivery."
    )


# Small footer
st.divider()

st.caption(
    "Built with Python, Pandas, scikit-learn, Plotly and Streamlit."
)