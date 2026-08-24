import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_test_email(candidate: dict, test: dict, config) -> bool:
    """
    Send skill test link to HOLD candidate via email.
    """
    email    = candidate.get('email')
    test_id  = test.get('test_id')
    name     = candidate.get('filename', 'Candidate').replace('.pdf', '').replace('.docx', '')
    job      = test.get('job_title', 'the role')

    if not email:
        print(f"⚠️ No email found for {name} — cannot send test")
        return False

    # Build test URL
    base_url = getattr(config, 'BASE_URL', 'http://localhost:5000')
    test_url = f"{base_url}/skill-test/{test_id}"

    subject = f"Skill Assessment — {job} Position"

    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: auto; padding: 20px;">

        <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 30px; border-radius: 12px; text-align: center;">
            <h1 style="color: white; margin: 0;">RecruitAI</h1>
            <p style="color: #e0e0e0; margin: 5px 0;">Skill Assessment Invitation</p>
        </div>

        <div style="padding: 30px; background: #f9f9f9; border-radius: 12px; margin-top: 20px;">
            <h2 style="color: #333;">Hello {name},</h2>

            <p style="color: #555; line-height: 1.6;">
                Thank you for applying for the <strong>{job}</strong> position.
                After reviewing your profile, we would like to invite you to complete
                a short skill assessment to help us better evaluate your candidacy.
            </p>

            <div style="background: white; border-left: 4px solid #667eea; padding: 15px; border-radius: 8px; margin: 20px 0;">
                <p style="margin: 0; color: #333;"><strong>Assessment Details:</strong></p>
                <p style="margin: 5px 0; color: #555;">• 5 multiple choice questions</p>
                <p style="margin: 5px 0; color: #555;">• Estimated time: 10 minutes</p>
                <p style="margin: 5px 0; color: #555;">• Pass mark: 60% (3 out of 5)</p>
                <p style="margin: 5px 0; color: #555;">• Topics: {', '.join(test.get('skills_tested', [])[:4])}</p>
            </div>

            <div style="text-align: center; margin: 30px 0;">
                <a href="{test_url}"
                   style="background: linear-gradient(135deg, #667eea, #764ba2);
                          color: white; padding: 14px 32px; border-radius: 8px;
                          text-decoration: none; font-weight: bold; font-size: 16px;">
                    Start Skill Assessment
                </a>
            </div>

            <p style="color: #888; font-size: 13px; text-align: center;">
                If the button doesn't work, copy this link:<br>
                <a href="{test_url}" style="color: #667eea;">{test_url}</a>
            </p>
        </div>

        <p style="color: #aaa; font-size: 12px; text-align: center; margin-top: 20px;">
            This is an automated message from RecruitAI.
        </p>
    </body>
    </html>
    """

    try:
        msg                    = MIMEMultipart('alternative')
        msg['Subject']         = subject
        msg['From']            = config.MAIL_USERNAME
        msg['To']              = email
        msg.attach(MIMEText(html, 'html'))

        with smtplib.SMTP(config.MAIL_SERVER, config.MAIL_PORT) as server:
            server.starttls()
            server.login(config.MAIL_USERNAME, config.MAIL_PASSWORD)
            server.sendmail(config.MAIL_USERNAME, email, msg.as_string())

        print(f"✅ Skill test email sent → {email}")
        return True

    except Exception as e:
        print(f"❌ Failed to send test email to {email}: {e}")
        return False