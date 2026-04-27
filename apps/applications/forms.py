from django import forms
from .models import Application, Document


class ApplicationForm(forms.ModelForm):
    class Meta:
        model = Application
        fields = [
            'full_name', 'dob', 'email', 'phone',
            'aadhaar_number', 'bank_account', 'address',
            'annual_income', 'academic_percentage'
        ]
        widgets = {
            'dob': forms.DateInput(attrs={'type': 'date'}),
            'address': forms.Textarea(attrs={'rows': 3}),
        }

    def clean_aadhaar_number(self):
        aadhaar = self.cleaned_data.get('aadhaar_number', '')
        if not aadhaar.isdigit() or len(aadhaar) != 12:
            raise forms.ValidationError("Aadhaar must be exactly 12 digits.")
        return aadhaar

    def clean_academic_percentage(self):
        pct = self.cleaned_data.get('academic_percentage')
        if pct < 0 or pct > 100:
            raise forms.ValidationError("Percentage must be between 0 and 100.")
        return pct


class DocumentUploadForm(forms.ModelForm):
    def __init__(self, *args, application=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.application = application

    class Meta:
        model = Document
        fields = ['doc_type', 'file']

    def clean_doc_type(self):
        doc_type = self.cleaned_data.get('doc_type')
        if (
            self.application
            and doc_type
            and self.application.documents.filter(doc_type=doc_type).exists()
        ):
            raise forms.ValidationError(
                "This document type has already been uploaded for the application."
            )
        return doc_type

    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            allowed = ['image/jpeg', 'image/png', 'application/pdf']
            if file.content_type not in allowed:
                raise forms.ValidationError("Only JPG, PNG, or PDF files are allowed.")
            if file.size > 5 * 1024 * 1024:  # 5MB limit
                raise forms.ValidationError("File size must be under 5MB.")
        return file
