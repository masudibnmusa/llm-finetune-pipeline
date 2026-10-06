from data_pipeline.cleaning.pii_scrubber import scrub_record, scrub_text


def test_email():
    assert scrub_text("Contact john.doe@example.com now") == "Contact [EMAIL] now"


def test_ssn():
    assert "[SSN]" in scrub_text("SSN: 123-45-6789")


def test_phone():
    assert "[PHONE]" in scrub_text("Call 555-123-4567 today")


def test_ip():
    assert "[IP_ADDRESS]" in scrub_text("Server at 192.168.1.1")


def test_record_skips_domain_key():
    rec = {"domain": "legal", "text": "mail me at a@b.com"}
    out = scrub_record(rec)
    assert out["domain"] == "legal"
    assert "[EMAIL]" in out["text"]