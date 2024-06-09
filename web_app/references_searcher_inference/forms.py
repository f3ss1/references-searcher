from django import forms


class PredictionForm(forms.Form):
    title = forms.CharField(widget=forms.TextInput(attrs={"placeholder": "Title"}))
    abstract = forms.CharField(widget=forms.Textarea(attrs={"placeholder": "Abstract"}))
