from django.http import HttpResponse
from django.contrib.admin.views.decorators import staff_member_required
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from apps.applications.models import Application
import csv


@staff_member_required
def export_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="applications.csv"'
    writer = csv.writer(response)
    writer.writerow(['Name', 'Email', 'Status', 'Fraud Score', 'Submitted At'])
    for app in Application.objects.all():
        writer.writerow([app.full_name, app.email, app.status, app.fraud_score, app.submitted_at])
    return response


@staff_member_required
def export_pdf(request):
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="fraud_report.pdf"'
    p = canvas.Canvas(response, pagesize=A4)
    p.setFont("Helvetica-Bold", 14)
    p.drawString(200, 800, "Scholarship Fraud Detection Report")
    p.setFont("Helvetica", 11)
    y = 760
    for app in Application.objects.all().order_by('-fraud_score')[:50]:
        p.drawString(50, y, f"{app.full_name} | {app.status} | Score: {app.fraud_score:.2f}")
        y -= 20
        if y < 50:
            p.showPage()
            y = 800
    p.save()
    return response