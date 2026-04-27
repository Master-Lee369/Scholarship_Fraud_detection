from apps.applications.models import Application
from .models import FraudFlag

def check_duplicates(application):
    duplicate_flag_types = [
        'duplicate_phone',
        'duplicate_email',
        'duplicate_aadhaar',
        'duplicate_bank',
        'bulk_submission',
        'ip_reuse',
    ]
    FraudFlag.objects.filter(
        application=application,
        flag_type__in=duplicate_flag_types,
    ).delete()

    flags = []

    # Check phone
    if Application.objects.filter(phone=application.phone).exclude(id=application.id).exists():
        flags.append(FraudFlag(
            application=application,
            flag_type='duplicate_phone',
            description=f"Phone {application.phone} reused in another application.",
            severity='high'
        ))

    # Check email
    if Application.objects.filter(email=application.email).exclude(id=application.id).exists():
        flags.append(FraudFlag(
            application=application,
            flag_type='duplicate_email',
            description=f"Email {application.email} reused.",
            severity='high'
        ))

    # Check Aadhaar
    if Application.objects.filter(aadhaar_number=application.aadhaar_number).exclude(id=application.id).exists():
        flags.append(FraudFlag(
            application=application,
            flag_type='duplicate_aadhaar',
            description=f"Aadhaar {application.aadhaar_number} already used.",
            severity='high'
        ))

    # Check bank account
    if Application.objects.filter(bank_account=application.bank_account).exclude(id=application.id).exists():
        flags.append(FraudFlag(
            application=application,
            flag_type='duplicate_bank',
            description=f"Bank account {application.bank_account} reused.",
            severity='medium'
        ))

    if (
        application.ip_address
        and Application.objects.filter(ip_address=application.ip_address).exclude(id=application.id).exists()
    ):
        flags.append(FraudFlag(
            application=application,
            flag_type='ip_reuse',
            description=f"IP address {application.ip_address} has been used before.",
            severity='medium'
        ))

    # Bulk submission from same IP (more than 3 in 1 hour)
    from django.utils import timezone
    from datetime import timedelta
    recent_ip_count = Application.objects.filter(
        ip_address=application.ip_address,
        submitted_at__gte=timezone.now() - timedelta(hours=1)
    ).exclude(id=application.id).count()

    if recent_ip_count >= 3:
        flags.append(FraudFlag(
            application=application,
            flag_type='bulk_submission',
            description=f"IP {application.ip_address} submitted {recent_ip_count+1} apps in 1 hour.",
            severity='high'
        ))

    FraudFlag.objects.bulk_create(flags)
    return flags
