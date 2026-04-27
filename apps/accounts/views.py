from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from apps.applications.models import Application
from apps.fraud_detection.models import FraudFlag
from django.db.models import Avg

from .forms import CustomUserCreationForm


def register_view(request):
    if request.user.is_authenticated:
        return redirect("profile")

    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("application_status")
    else:
        form = CustomUserCreationForm()

    return render(request, "accounts/register.html", {"form": form})


@login_required
def profile_redirect(request):
    if request.user.is_staff:
        return redirect("admin_dashboard")
    return redirect("application_status")


@login_required
def logout_view(request):
    logout(request)
    return redirect("login")


@staff_member_required
def admin_dashboard(request):
    stats = {
        'total': Application.objects.count(),
        'genuine': Application.objects.filter(status='genuine').count(),
        'suspicious': Application.objects.filter(status='suspicious').count(),
        'fake': Application.objects.filter(status='fake').count(),
        'pending': Application.objects.filter(status='pending').count(),
        'avg_fraud_score': Application.objects.aggregate(avg=Avg('fraud_score'))['avg'] or 0,
    }
    flagged = Application.objects.filter(status='suspicious').order_by('-fraud_score')[:10]
    recent_flags = FraudFlag.objects.order_by('-created_at')[:20]

    return render(request, 'admin_panel/dashboard.html', {
        'stats': stats,
        'flagged': flagged,
        'recent_flags': recent_flags,
    })


@staff_member_required
def review_application(request, pk):
    application = get_object_or_404(Application, pk=pk)
    flags = application.flags.all()
    docs = application.documents.all()

    if request.method == 'POST':
        action = request.POST.get('action')
        notes = request.POST.get('notes', '')
        if action in ['approved', 'rejected']:
            application.status = action
            application.admin_notes = notes
            application.save()
        return redirect('admin_dashboard')

    return render(request, 'admin_panel/review.html', {
        'application': application,
        'flags': flags,
        'docs': docs,
    })
