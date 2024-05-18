from django import forms


class PredictionForm(forms.Form):
    title = forms.CharField(max_length=2000, widget=forms.TextInput(attrs={"placeholder": "Title"}))
    abstract = forms.CharField(widget=forms.Textarea(attrs={"placeholder": "Abstract"}))
