from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import ApplicationForm, DocumentUploadForm
from .models import Application, Document
from apps.verification.ocr_engine import extract_text, detect_tampering, cross_verify_documents
from apps.fraud_detection.duplicate_checker import check_duplicates
from apps.fraud_detection.ml_scorer import classify_application
from apps.fraud_detection.models import FraudFlag


@login_required
def submit_application(request):
    # Prevent duplicate submissions
    if Application.objects.filter(user=request.user).exists():
        messages.warning(request, "You have already submitted an application.")
        return redirect('application_status')

    if request.method == 'POST':
        form = ApplicationForm(request.POST)
        if form.is_valid():
            application = form.save(commit=False)
            application.user = request.user
            application.ip_address = get_client_ip(request)
            application.save()
            messages.success(request, "Application submitted. Please upload your documents.")
            return redirect('upload_documents', pk=application.pk)
    else:
        form = ApplicationForm()

    return render(request, 'applications/submit.html', {'form': form})


@login_required
def upload_documents(request, pk):
    application = get_object_or_404(Application, pk=pk, user=request.user)

    if request.method == 'POST':
        form = DocumentUploadForm(request.POST, request.FILES, application=application)
        if form.is_valid():
            doc = form.save(commit=False)
            doc.application = application
            doc.save()

            # Run OCR
            doc.extracted_text = extract_text(doc.file.path)
            doc.tamper_score = detect_tampering(doc.file.path)
            doc.is_verified = doc.tamper_score < 0.15
            doc.save()

            if application.has_all_required_documents():
                check_duplicates(application)
                FraudFlag.objects.filter(application=application, flag_type='doc_tamper').delete()
                for issue in cross_verify_documents(application):
                    FraudFlag.objects.create(
                        application=application,
                        flag_type='doc_tamper',
                        description=issue,
                        severity='medium'
                    )
                classify_application(application)
                messages.success(request, "Application fully submitted and verified.")
                return redirect('application_status')

            messages.success(request, f"Document '{doc.doc_type}' uploaded successfully.")
    else:
        form = DocumentUploadForm(application=application)

    uploaded = application.documents.all()
    return render(request, 'applications/upload_documents.html', {
        'form': form,
        'application': application,
        'uploaded': uploaded,
        'missing_docs': application.missing_document_types(),
    })


def get_client_ip(request):
    x_forwarded = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded:
        return x_forwarded.split(',')[0]
    return request.META.get('REMOTE_ADDR')


@login_required
def application_status(request):
    application = Application.objects.filter(user=request.user).first()
    return render(request, 'applications/status.html', {'application': application})
