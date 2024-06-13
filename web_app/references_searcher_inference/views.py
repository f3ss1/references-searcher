from django.shortcuts import render
from django.http import JsonResponse
from celery.result import AsyncResult

from .tasks import make_predictions
from .forms import PredictionForm

from references_searcher import logger
from references_searcher.models.inferencer import ReferencePrediction


def predict_references(request):
    if request.method == "POST":
        form = PredictionForm(request.POST)
        if form.is_valid():
            logger.info("Got a valid form, converting to pandas dataframe.")

            title = form.cleaned_data["title"]
            abstract = form.cleaned_data["abstract"]
            model_input = {"title": [title], "abstract": [abstract]}

            logger.info("Making predictions for the request.")
            predictions_task = make_predictions.delay(model_input)
            request.session["predictions_task_id"] = predictions_task.id
            logger.info(type(predictions_task.id))
            logger.info(predictions_task.id)
            logger.info("Here should be a page!")
            return render(request, "references_searcher_inference/wait.html")
    else:
        # request.session["predictions_task_id"] = None
        existing_predictions_task_id = request.session.get("predictions_task_id")
        logger.info(f"Celery task: {existing_predictions_task_id}")
        if existing_predictions_task_id is None:
            logger.info("Making form.")
            form = PredictionForm()
            return render(request, "references_searcher_inference/form.html", {"form": form})
        else:
            existing_predictions_task = AsyncResult(existing_predictions_task_id)
            logger.info("Existing task found.")
            if existing_predictions_task.state == "SUCCESS":
                logger.info("Task finished, rendering results.")
                request.session["predictions_task_id"] = None
                celery_result = [ReferencePrediction.from_dict(x) for x in existing_predictions_task.get()]
                return render(
                    request,
                    "references_searcher_inference/results.html",
                    {"predictions": celery_result},
                )
            else:
                logger.info(f"Task is still in progress, status: {existing_predictions_task.state}")
                return render(request, "references_searcher_inference/wait.html")


def task_status(request):
    task_id = request.session.get("predictions_task_id")
    if not task_id:
        return JsonResponse({"ready": False})

    task = AsyncResult(task_id)
    if task.state == "SUCCESS":
        return JsonResponse({"ready": True})
    else:
        return JsonResponse({"ready": False})
