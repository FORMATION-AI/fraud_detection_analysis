from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html", results=None, error_message=None)

    try:
        data = CustomData(
            amount=float(request.form.get("amount")),
            customer_age=float(request.form.get("customer_age")),
            minute_of_day=float(request.form.get("minute_of_day")),
            to_acc_volume=float(request.form.get("to_acc_volume")),
            session_duration=float(request.form.get("session_duration")),
            hour_of_day=int(request.form.get("hour_of_day")),
            day_of_week=int(request.form.get("day_of_week")),
        )
    except (TypeError, ValueError):
        return render_template(
            "home.html",
            results=None,
            error_message="Invalid input supplied. Please check values.",
        )

    pred_df = data.get_data_as_data_frame()
    predict_pipeline = PredictPipeline()

    try:
        results = predict_pipeline.predict(pred_df)
    except Exception:
        return render_template(
            "home.html",
            results=None,
            error_message="Prediction failed. Check logs for details.",
        )

    result_text = f"Prediction executed. Model output: {int(results[0])}"
    return render_template("home.html", results=result_text, error_message=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
