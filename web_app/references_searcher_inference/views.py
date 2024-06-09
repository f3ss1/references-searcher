import pandas as pd

from django.shortcuts import render
from .apps import ReferenceSearcherInferenceConfig
from .forms import PredictionForm


def predict_references(request):
    if request.method == "POST":
        form = PredictionForm(request.POST)
        if form.is_valid():
            title = form.cleaned_data["title"]
            abstract = form.cleaned_data["abstract"]
            model_input = pd.DataFrame({"title": [title], "abstract": [abstract]})
            predictions = ReferenceSearcherInferenceConfig.inferencer.predict(model_input)[0]
            return render(
                request,
                "references_searcher_inference/results.html",
                {"form": form, "predictions": predictions},
            )
    else:
        form = PredictionForm()
    return render(request, "references_searcher_inference/form.html", {"form": form})
