"""
modules/interview_scheduler.py
──────────────────────────────
iCal-based Interview Scheduler for RecruitAI.

Fixes applied:
  1. Interview slots start from NEXT WEEK (not tomorrow)
  2. Finds ONE non-busy slot from recruiter's calendar for ALL candidates of same role
  3. All shortlisted candidates for same job get same date/time
  4. Mode fixed to In-Person
  5. Candidate name extracted from resume text (not filename)
  6. Job title cleaned (removes timestamp suffix)
"""

import os
import uuid
import json
import re
import smtplib
import requests
from datetime import datetime, timedelta, timezone, date
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from jinja2 import Template

try:
    from icalendar import Calendar
    ICAL_AVAILABLE = True
except ImportError:
    ICAL_AVAILABLE = False
    print("icalendar not installed. Run: pip install icalendar requests")

# ── Configuration ──────────────────────────────────────────────────────────
WORK_START_HOUR = 9    # 9 AM
WORK_END_HOUR   = 18   # 6 PM
DURATION_MINS   = 60   # 1-hour interview slots
INTERVIEW_MODE  = "In-Person"
INTERVIEW_LOCATION = "Company Office"

# Interview happens NEXT WEEK minimum
MIN_DAYS_AHEAD  = 7    # at least 7 days from today
MAX_DAYS_AHEAD  = 21   # search up to 3 weeks ahead
# ──────────────────────────────────────────────────────────────────────────


class InterviewScheduler:

    def __init__(self, config):
        self.config = config

        base = config.BASE_DIR
        self.slots_folder         = os.path.join(base, "interviews", "slots")
        self.invitations_folder   = os.path.join(base, "interviews", "invitations")
        self.confirmations_folder = os.path.join(base, "interviews", "confirmations")

        for d in [self.slots_folder, self.invitations_folder, self.confirmations_folder]:
            os.makedirs(d, exist_ok=True)

        self.smtp_server   = os.environ.get("SMTP_SERVER",   "smtp.gmail.com")
        self.smtp_port     = int(os.environ.get("SMTP_PORT", 587))
        self.smtp_username = os.environ.get("SMTP_USERNAME", "")
        self.smtp_password = os.environ.get("SMTP_PASSWORD", "")
        self.from_email    = os.environ.get("FROM_EMAIL", self.smtp_username or "noreply@recruitai.com")
        self.ical_url      = os.environ.get("RECRUITER_ICAL_URL", "")

        # Company info — set these in your .env file
        self.company_name     = os.environ.get("COMPANY_NAME",     "Our Organization")
        self.company_address  = os.environ.get("COMPANY_ADDRESS",  "Company Office")
        self.company_city     = os.environ.get("COMPANY_CITY",     "")
        self.company_phone    = os.environ.get("COMPANY_PHONE",    "")

        # Full location string shown in email and calendar
        parts = [self.company_name, self.company_address]
        if self.company_city:
            parts.append(self.company_city)
        self.full_location = ", ".join(p for p in parts if p)

        print("Interview Scheduler (iCal mode) initialised")
        print(f"Company: {self.company_name} | Location: {self.full_location}")
        if self.ical_url:
            print("iCal URL found — will find next-week free slot from Google Calendar")
        else:
            print("RECRUITER_ICAL_URL not set — will pick next Monday 10 AM as default")

    # ══════════════════════════════════════════════════════════════════
    #  HELPER: Clean job title (remove timestamp suffix)
    # ══════════════════════════════════════════════════════════════════

    def _clean_job_title(self, title: str) -> str:
        """
        Remove timestamp suffixes from job titles.
        'Software Developer Backend Python 20260424' → 'Software Developer Backend Python'
        'jd_python_developer_20260424_104828'        → 'Python Developer'
        """
        # Remove trailing 8-digit date like 20260424
        title = re.sub(r'\s+\d{8}$', '', title.strip())
        # Remove trailing timestamp like _20260424_104828
        title = re.sub(r'_\d{8}_\d{6}$', '', title)
        # Remove jd_ prefix
        title = re.sub(r'^jd_', '', title)
        # Replace underscores with spaces and title-case
        title = title.replace('_', ' ').strip().title()
        return title

    # ══════════════════════════════════════════════════════════════════
    #  HELPER: Extract real candidate name from resume text
    # ══════════════════════════════════════════════════════════════════

    def _extract_candidate_name(self, filename: str, resume_text: str = "") -> str:
        """
        Try to get the real name from resume text.
        Falls back to cleaned filename if not found.
        """
        if resume_text:
            lines = [l.strip() for l in resume_text.split('\n')
                     if l.strip() and len(l.strip()) > 1]

            for line in lines[:10]:  # Name is usually in first 10 lines
                # Skip lines that look like emails, phones, addresses, headers
                if any(skip in line.lower() for skip in [
                    '@', 'http', 'www', 'phone', 'email', 'address',
                    'resume', 'curriculum', 'objective', 'summary',
                    'linkedin', 'github', '+91', 'mobile'
                ]):
                    continue
                # Skip lines with numbers (phone, dates)
                if re.search(r'\d{4,}', line):
                    continue
                # Name lines are usually 2-4 words, each capitalized
                words = line.split()
                if 2 <= len(words) <= 4:
                    if all(w[0].isupper() for w in words if w.isalpha()):
                        # Looks like a proper name
                        return ' '.join(words)

        # Fallback: clean the filename
        name = filename.replace('.pdf', '').replace('.docx', '').replace('.txt', '')
        # Remove UUID prefix (8 chars + underscore)
        name = re.sub(r'^[a-f0-9]{8}_', '', name)
        # Replace underscores/hyphens with spaces
        name = name.replace('_', ' ').replace('-', ' ')
        # Remove common resume words
        name = re.sub(r'\b(resume|cv|updated|final|new|1|2|3)\b', '', name, flags=re.I)
        name = ' '.join(name.split()).title()
        return name or "Candidate"

    # ══════════════════════════════════════════════════════════════════
    #  HELPER: Extract company name + location from JD text
    # ══════════════════════════════════════════════════════════════════

    def _extract_company_from_jd(self, jd_text: str) -> tuple:
        """
        Parse company name and location directly from the JD text.

        Handles patterns like:
          Company: ABC Corp
          Location: Thrissur, Kerala
          About ABC Corp
          We are ABC Corp, located in Thrissur
        """
        company_name = self.company_name   # default from env
        location     = self.full_location  # default from env

        if not jd_text:
            return company_name, location

        lines = jd_text.split('\n')

        # Pattern 1: explicit "Company: XYZ" line
        for line in lines:
            m = re.search(
                r'(?:company|organisation|organization|employer)\s*[:\-]\s*(.+)',
                line, re.IGNORECASE
            )
            if m:
                val = m.group(1).strip().strip('.,')
                if 3 < len(val) < 80:
                    company_name = val
                    break

        # Pattern 2: explicit "Location: XYZ" line
        for line in lines:
            m = re.search(
                r'(?:location|office|address|venue|place)\s*[:\-]\s*(.+)',
                line, re.IGNORECASE
            )
            if m:
                val = m.group(1).strip().strip('.,')
                if 3 < len(val) < 120:
                    location = val
                    break

        # Pattern 3: "About <Company Name>" heading
        if company_name == self.company_name:
            for line in lines:
                m = re.match(
                    r'^about\s+(?!us\b|the\s+role|this\s+role)(.+)',
                    line.strip(), re.IGNORECASE
                )
                if m:
                    val = m.group(1).strip().strip('.,')
                    if 3 < len(val) < 80:
                        company_name = val
                        break

        # Pattern 4: "located in <City, State>"
        if location == self.full_location:
            m = re.search(
                r'located\s+in\s+([A-Za-z\s,]+?)(?:\.|,\s*\n|\n|$)',
                jd_text, re.IGNORECASE
            )
            if m:
                val = m.group(1).strip().strip('.,')
                if 3 < len(val) < 120:
                    location = val

        return (company_name or "Our Organization",
                location     or "Company Office")

    # ══════════════════════════════════════════════════════════════════
    #  iCAL: fetch busy slots
    # ══════════════════════════════════════════════════════════════════

    def _fetch_busy_slots(self, ical_url: str) -> list:
        """
        Download iCal feed and return list of (start, end) busy tuples.
        """
        if not ICAL_AVAILABLE or not ical_url:
            return []

        try:
            resp = requests.get(ical_url, timeout=10)
            resp.raise_for_status()
            cal  = Calendar.from_ical(resp.content)

            busy     = []
            now      = datetime.now(timezone.utc)
            scan_end = now + timedelta(days=MAX_DAYS_AHEAD + 2)

            for component in cal.walk():
                if component.name != "VEVENT":
                    continue
                dtstart = component.get("dtstart")
                dtend   = component.get("dtend")
                if not dtstart or not dtend:
                    continue

                start = dtstart.dt
                end   = dtend.dt

                # Skip all-day events
                if not hasattr(start, "hour"):
                    continue

                # Make timezone-aware
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)

                if start > scan_end or end < now:
                    continue

                busy.append((start, end))

            print(f"iCal: {len(busy)} busy period(s) found")
            return busy

        except Exception as e:
            print(f"iCal fetch error: {e} — treating calendar as empty")
            return []

    # ══════════════════════════════════════════════════════════════════
    #  CORE: Find ONE best slot from next week onwards
    # ══════════════════════════════════════════════════════════════════

    def _get_already_booked_slots(self) -> list:
        """
        Read all existing slot groups saved by previous screenings.
        Returns list of (start, end) tuples so we don't double-book the same time.
        Each job role gets its own unique slot.
        """
        booked = []
        if not os.path.exists(self.slots_folder):
            return booked

        for file in os.listdir(self.slots_folder):
            if not file.startswith("group_") or not file.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.slots_folder, file)) as f:
                    group = json.load(f)
                for slot in group.get("slots", []):
                    # Only count slots that are still in the future
                    try:
                        slot_dt = datetime.fromisoformat(
                            f"{slot['date']}T{slot['start_time']}"
                        ).replace(tzinfo=timezone.utc)
                        slot_end = datetime.fromisoformat(
                            f"{slot['date']}T{slot['end_time']}"
                        ).replace(tzinfo=timezone.utc)
                        if slot_dt > datetime.now(timezone.utc):
                            booked.append((slot_dt, slot_end))
                    except Exception:
                        pass
            except Exception:
                pass

        print(f"Already booked slots (other roles): {len(booked)}")
        return booked

    def find_best_interview_slot(self, recruiter_email: str) -> dict | None:
        """
        Find ONE free slot starting from NEXT WEEK.
        Checks BOTH:
          1. Recruiter's Google Calendar busy times (iCal feed)
          2. Already-booked slots from other job role screenings
        So each job role gets a different date/time automatically.

        Example:
          Backend Developer  → Monday  04 May 10:00 AM
          Frontend Developer → Monday  04 May 11:00 AM  (next free after backend)
          Data Analyst       → Tuesday 05 May 10:00 AM  (if Monday is full)
        """
        # Busy from Google Calendar
        calendar_busy = self._fetch_busy_slots(self.ical_url) if self.ical_url else []

        # Busy from already-scheduled interviews (other roles)
        role_busy = self._get_already_booked_slots()

        # Combine both sources
        all_busy = calendar_busy + role_busy

        now      = datetime.now(timezone.utc)
        slot_dur = timedelta(minutes=DURATION_MINS)

        # Start from next Monday (at least MIN_DAYS_AHEAD away)
        days_until_monday = (7 - now.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7
        if days_until_monday < MIN_DAYS_AHEAD:
            days_until_monday = MIN_DAYS_AHEAD
            target = now + timedelta(days=days_until_monday)
            while target.weekday() != 0:
                target += timedelta(days=1)
        else:
            target = now + timedelta(days=days_until_monday)

        target = target.replace(hour=WORK_START_HOUR, minute=0, second=0, microsecond=0)

        # Preferred times: 10 AM first, then 11 AM, then rest
        preferred_hours = [10, 11, 9, 14, 15, 16]

        days_checked = 0
        while days_checked < (MAX_DAYS_AHEAD - MIN_DAYS_AHEAD + 7):
            # Skip weekends
            if target.weekday() >= 5:
                target += timedelta(days=1)
                target  = target.replace(hour=WORK_START_HOUR, minute=0,
                                         second=0, microsecond=0)
                days_checked += 1
                continue

            for hour in preferred_hours:
                slot_start = target.replace(hour=hour, minute=0,
                                            second=0, microsecond=0)
                slot_end   = slot_start + slot_dur

                if slot_end.hour > WORK_END_HOUR:
                    continue

                # Check against Google Calendar AND other role bookings
                overlap = any(
                    slot_start < b_end and slot_end > b_start
                    for b_start, b_end in all_busy
                )

                if not overlap:
                    slot_id = str(uuid.uuid4())
                    print(f"✅ Slot found for this role: "
                          f"{slot_start.strftime('%A %d %B %Y at %I:%M %p')}")
                    return {
                        "id":              slot_id,
                        "date":            slot_start.strftime("%Y-%m-%d"),
                        "start_time":      slot_start.strftime("%H:%M"),
                        "end_time":        slot_end.strftime("%H:%M"),
                        "duration":        DURATION_MINS,
                        "mode":            INTERVIEW_MODE,
                        "location":        self.full_location,
                        "company_name":    self.company_name,
                        "recruiter_email": recruiter_email,
                        "status":          "available",
                        "created_at":      datetime.now().isoformat(),
                        "booked_by":       None,
                        "booking_time":    None,
                        "display_date":    slot_start.strftime("%A, %d %B %Y"),
                        "display_time":    slot_start.strftime("%I:%M %p"),
                        "display_end":     slot_end.strftime("%I:%M %p"),
                    }

            # All preferred hours on this day are taken — move to next day
            target += timedelta(days=1)
            target  = target.replace(hour=WORK_START_HOUR, minute=0,
                                     second=0, microsecond=0)
            days_checked += 1

        print("⚠️ No free slot found — using fallback")
        return self._fallback_slot(recruiter_email)

    def _fallback_slot(self, recruiter_email: str) -> dict:
        """Fallback: next Monday at 10 AM if nothing found in calendar."""
        now = datetime.now(timezone.utc)
        days_ahead = (7 - now.weekday()) % 7
        if days_ahead < 7:
            days_ahead += 7
        monday = (now + timedelta(days=days_ahead)).replace(
            hour=10, minute=0, second=0, microsecond=0
        )
        slot_end = monday + timedelta(minutes=DURATION_MINS)
        slot_id  = str(uuid.uuid4())
        return {
            "id":              slot_id,
            "date":            monday.strftime("%Y-%m-%d"),
            "start_time":      monday.strftime("%H:%M"),
            "end_time":        slot_end.strftime("%H:%M"),
            "duration":        DURATION_MINS,
            "mode":            INTERVIEW_MODE,
            "location":        self.full_location,
            "company_name":    self.company_name,
            "recruiter_email": recruiter_email,
            "status":          "available",
            "created_at":      datetime.now().isoformat(),
            "booked_by":       None,
            "booking_time":    None,
            "display_date":    monday.strftime("%A, %d %B %Y"),
            "display_time":    monday.strftime("%I:%M %p"),
            "display_end":     slot_end.strftime("%I:%M %p"),
        }

    # ══════════════════════════════════════════════════════════════════
    #  create_slots — now finds ONE shared slot for all candidates
    # ══════════════════════════════════════════════════════════════════

    def create_slots(self, recruiter_email: str, slots_data=None) -> str:
        """
        Find ONE best free slot from next week and create a slot group.
        All shortlisted candidates for this role share the same slot.
        """
        slot_group_id = str(uuid.uuid4())

        best_slot = self.find_best_interview_slot(recruiter_email)

        if best_slot:
            best_slot["group_id"] = slot_group_id
            slots = [best_slot]
        else:
            slots = []

        # Save each slot
        for slot in slots:
            with open(os.path.join(self.slots_folder, f"{slot['id']}.json"), "w") as f:
                json.dump(slot, f, indent=2)

        # Save group
        group = {
            "group_id":        slot_group_id,
            "recruiter_email": recruiter_email,
            "slots":           slots,
            "created_at":      datetime.now().isoformat(),
            "source":          "ical" if self.ical_url else "fallback",
        }
        with open(os.path.join(self.slots_folder, f"group_{slot_group_id}.json"), "w") as f:
            json.dump(group, f, indent=2)

        if slots:
            s = slots[0]
            print(f"Slot group created — Interview: {s.get('display_date','?')} at {s.get('display_time','?')} [{s['mode']}]")
        else:
            print("Slot group created — 0 slots (calendar fully busy)")

        return slot_group_id

    # ══════════════════════════════════════════════════════════════════
    #  send_interview_invites — now sends CONFIRMED time (not selection)
    # ══════════════════════════════════════════════════════════════════

    def send_interview_invites(self, candidate_emails: list, candidate_names: list,
                               job_title: str, slot_group_id: str, base_url: str,
                               resume_texts: list = None,
                               job_data: dict = None) -> list:
        """
        Send interview confirmation emails to all shortlisted candidates.
        All candidates get the SAME date/time — no slot selection needed.
        Email includes .ics calendar attachment.
        Company name and location extracted from the JD text automatically.
        """
        # Clean job title
        clean_title = self._clean_job_title(job_title)

        # Extract company info from JD text
        jd_text = ""
        if job_data:
            jd_text = job_data.get("description", "")
        company_name, company_location = self._extract_company_from_jd(jd_text)
        print(f"Company from JD: {company_name} | Location: {company_location}")

        # Get the slot
        slots = self.get_available_slots(slot_group_id)
        if not slots:
            print("⚠️ No slots available in group — cannot send invites")
            return []

        slot = slots[0]  # All candidates get same slot

        invitations = []

        for i, email in enumerate(candidate_emails):
            # Get real name
            raw_name     = candidate_names[i] if i < len(candidate_names) else ""
            resume_text  = resume_texts[i] if resume_texts and i < len(resume_texts) else ""
            display_name = self._extract_candidate_name(raw_name, resume_text)

            token    = str(uuid.uuid4())
            inv_id   = str(uuid.uuid4())

            invitation = {
                "id":              inv_id,
                "token":           token,
                "candidate_email": email,
                "candidate_name":  display_name,
                "job_title":       clean_title,
                "slot_group_id":   slot_group_id,
                "slot_id":         slot["id"],
                "status":          "sent",
                "sent_at":         datetime.now().isoformat(),
                "expires_at":      (datetime.now() + timedelta(days=14)).isoformat(),
            }

            with open(os.path.join(self.invitations_folder, f"{token}.json"), "w") as f:
                json.dump(invitation, f, indent=2)

            # Send confirmed interview email (not slot selection)
            self._send_confirmed_interview_email(
                invitation, slot, clean_title,
                company_name=company_name,
                company_location=company_location,
            )
            invitations.append(invitation)
            print(f"Interview invite sent → {email} ({display_name})")

        return invitations

    # ══════════════════════════════════════════════════════════════════
    #  CONFIRMED INTERVIEW EMAIL (replaces slot-selection email)
    # ══════════════════════════════════════════════════════════════════

    def _send_confirmed_interview_email(self, invitation: dict, slot: dict,
                                         job_title: str, company_name: str = "",
                                         company_location: str = ""):
        """
        Send a confirmed interview email — date and time already decided.
        Includes .ics attachment so candidate adds it to calendar in one click.
        """
        candidate_name  = invitation["candidate_name"]
        candidate_email = invitation["candidate_email"]
        recruiter_email = slot["recruiter_email"]

        # Use JD-extracted values, fall back to instance defaults
        org_name     = company_name     or self.company_name     or "Our Organization"
        org_location = company_location or self.full_location    or "Company Office"
        # Update the slot location so .ics also has the right venue
        slot["location"] = org_location

        display_date = slot.get("display_date") or datetime.fromisoformat(
            f"{slot['date']}T{slot['start_time']}"
        ).strftime("%A, %d %B %Y")

        display_time = slot.get("display_time") or slot["start_time"]
        display_end  = slot.get("display_end")  or slot["end_time"]

        html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6f9; margin: 0; padding: 20px; }}
  .container {{ max-width: 600px; margin: 0 auto; background: #fff;
                border-radius: 12px; overflow: hidden;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
  .header {{ background: linear-gradient(135deg, #4f46e5, #7c3aed);
             padding: 36px 32px; text-align: center; }}
  .header h1 {{ color: #fff; margin: 0; font-size: 22px; font-weight: 700; }}
  .header p  {{ color: rgba(255,255,255,0.85); margin: 8px 0 0; font-size: 14px; }}
  .body {{ padding: 32px; }}
  .congrats {{ font-size: 17px; color: #1a202c; margin-bottom: 24px; line-height: 1.6; }}
  .details-card {{
    background: #f8f9ff; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 24px; margin: 24px 0;
  }}
  .details-card h3 {{ margin: 0 0 16px; color: #4f46e5; font-size: 15px; }}
  .detail-row {{ display: flex; margin-bottom: 12px; align-items: flex-start; }}
  .detail-label {{ font-weight: 600; color: #4a5568; width: 130px; flex-shrink: 0; font-size: 13px; }}
  .detail-value {{ color: #1a202c; font-size: 13px; }}
  .highlight {{ color: #4f46e5; font-weight: 700; font-size: 15px; }}
  .ics-note {{
    background: #e8f5e9; border-left: 4px solid #43a047;
    border-radius: 6px; padding: 14px 16px; margin: 20px 0;
    font-size: 13px; color: #2e7d32;
  }}
  .footer {{ background: #f8fafc; padding: 20px 32px; text-align: center;
             font-size: 12px; color: #718096; border-top: 1px solid #e2e8f0; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>🎉 Interview Confirmed</h1>
    <p>{org_name} — Recruitment Team</p>
  </div>
  <div class="body">
    <div class="congrats">
      Dear <strong>{candidate_name}</strong>,<br><br>
      Congratulations! You have been shortlisted for the
      <strong>{job_title}</strong> position at <strong>{org_name}</strong>.
      Your interview has been scheduled — please find the details below.
    </div>

    <div class="details-card">
      <h3>📋 Interview Details</h3>
      <div class="detail-row">
        <span class="detail-label">Organisation</span>
        <span class="detail-value"><b>{org_name}</b></span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Position</span>
        <span class="detail-value">{job_title}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Date</span>
        <span class="detail-value highlight">📅 {display_date}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Time</span>
        <span class="detail-value highlight">🕙 {display_time} – {display_end}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Mode</span>
        <span class="detail-value">🏢 {slot.get('mode', 'In-Person')}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Venue</span>
        <span class="detail-value">📍 {org_location}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Duration</span>
        <span class="detail-value">{slot.get('duration', 60)} minutes</span>
      </div>
    </div>

    <div class="ics-note">
      📎 A calendar invite (.ics) is attached to this email.
      Open the attachment to add this interview directly to your
      Google Calendar, Outlook, or Apple Calendar with automatic reminders.
    </div>

    <p style="color:#4a5568;font-size:13px;line-height:1.6">
      Please reply to this email to confirm your attendance.<br>
      For any queries, contact us at
      <a href="mailto:{recruiter_email}">{recruiter_email}</a>.
    </p>

    <p style="color:#4a5568;font-size:13px">
      We look forward to meeting you!<br>
      <strong>HR Team, {org_name}</strong>
    </p>
  </div>
  <div class="footer">
    {org_name} &bull; {org_location} &bull;
    Powered by RecruitAI
  </div>
</div>
</body>
</html>
"""
        # Build confirmation dict for .ics generation
        confirmation = {
            "id":              invitation["id"],
            "candidate_name":  candidate_name,
            "candidate_email": candidate_email,
            "recruiter_email": recruiter_email,
            "job_title":       job_title,
        }
        ics = self._generate_ics(confirmation, slot)

        # Send to candidate
        self._send_email(
            to_email     = candidate_email,
            subject      = f"Interview Confirmed: {job_title} — {display_date} at {display_time}",
            html_content = html,
            ics_content  = ics,
        )

        # Notify recruiter — with .ics so it adds to their calendar automatically
        recruiter_html = f"""
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{ font-family:'Segoe UI',Arial,sans-serif; background:#f4f6f9;
          margin:0; padding:20px; }}
  .card {{ max-width:560px; margin:0 auto; background:#fff;
           border-radius:12px; overflow:hidden;
           box-shadow:0 4px 20px rgba(0,0,0,.08); }}
  .hdr  {{ background:linear-gradient(135deg,#1a56db,#0e4bbd);
           padding:28px 28px 20px; }}
  .hdr h2 {{ color:#fff; margin:0; font-size:18px; }}
  .hdr p  {{ color:rgba(255,255,255,.8); margin:6px 0 0; font-size:13px; }}
  .body {{ padding:28px; }}
  .row  {{ display:flex; margin-bottom:10px; font-size:13px; }}
  .lbl  {{ font-weight:600; color:#4a5568; width:120px; flex-shrink:0; }}
  .val  {{ color:#1a202c; }}
  .note {{ background:#ebf8ff; border-left:4px solid #3182ce;
           border-radius:6px; padding:12px 14px; margin-top:16px;
           font-size:12px; color:#2c5282; }}
</style>
</head>
<body>
<div class="card">
  <div class="hdr">
    <h2>📅 New Interview Scheduled — Action Required</h2>
    <p>RecruitAI has scheduled an interview on your behalf</p>
  </div>
  <div class="body">
    <div class="row"><span class="lbl">Candidate</span>
      <span class="val"><b>{candidate_name}</b> ({candidate_email})</span></div>
    <div class="row"><span class="lbl">Role</span>
      <span class="val">{job_title}</span></div>
    <div class="row"><span class="lbl">Date</span>
      <span class="val"><b>📅 {display_date}</b></span></div>
    <div class="row"><span class="lbl">Time</span>
      <span class="val"><b>🕙 {display_time} – {display_end}</b></span></div>
    <div class="row"><span class="lbl">Mode</span>
      <span class="val">{slot.get('mode','In-Person')}</span></div>
    <div class="row"><span class="lbl">Location</span>
      <span class="val">{org_location}</span></div>
    <div class="note">
      📎 <b>Open the attached .ics file</b> to add this interview to your
      Google Calendar automatically. The event will appear on
      <b>{display_date}</b> from <b>{display_time}</b> with reminders.
    </div>
    <p style="color:#718096;font-size:12px;margin-top:16px">
      The candidate has also been notified with the same details.<br>
      Scheduled automatically by RecruitAI.
    </p>
  </div>
</div>
</body>
</html>
"""
        self._send_email(
            to_email     = recruiter_email,
            subject      = f"📅 Interview Scheduled: {candidate_name} → {job_title} on {display_date}",
            html_content = recruiter_html,
            ics_content  = ics,
        )

    # ══════════════════════════════════════════════════════════════════
    #  .ICS GENERATOR
    # ══════════════════════════════════════════════════════════════════

    def _generate_ics(self, confirmation: dict, slot: dict) -> str:
        start_dt  = datetime.fromisoformat(f"{slot['date']}T{slot['start_time']}")
        end_dt    = datetime.fromisoformat(f"{slot['date']}T{slot['end_time']}")
        fmt       = "%Y%m%dT%H%M%SZ"
        now_str   = datetime.utcnow().strftime(fmt)
        start_str = start_dt.strftime(fmt)
        end_str   = end_dt.strftime(fmt)
        uid       = f"{confirmation['id']}@recruitai.com"
        summary   = f"Interview: {confirmation['job_title']}"
        location  = slot.get("location", INTERVIEW_LOCATION)
        desc      = (
            f"Interview for {confirmation['job_title']}.\\n"
            f"Candidate: {confirmation['candidate_name']} ({confirmation['candidate_email']})\\n"
            f"Mode: {slot.get('mode', INTERVIEW_MODE)}\\n"
            f"Location: {location}\\n"
            f"Scheduled via RecruitAI"
        )
        return (
            "BEGIN:VCALENDAR\r\n"
            "VERSION:2.0\r\n"
            "PRODID:-//RecruitAI//Interview Scheduler//EN\r\n"
            "METHOD:REQUEST\r\n"
            "BEGIN:VEVENT\r\n"
            f"UID:{uid}\r\n"
            f"DTSTAMP:{now_str}\r\n"
            f"DTSTART:{start_str}\r\n"
            f"DTEND:{end_str}\r\n"
            f"SUMMARY:{summary}\r\n"
            f"DESCRIPTION:{desc}\r\n"
            f"LOCATION:{location}\r\n"
            f"ORGANIZER:mailto:{confirmation['recruiter_email']}\r\n"
            f"ATTENDEE;RSVP=TRUE:mailto:{confirmation['candidate_email']}\r\n"
            f"ATTENDEE;RSVP=TRUE:mailto:{confirmation['recruiter_email']}\r\n"
            "STATUS:CONFIRMED\r\n"
            "BEGIN:VALARM\r\n"
            "TRIGGER:-PT1H\r\n"
            "ACTION:DISPLAY\r\n"
            "DESCRIPTION:Interview reminder — 1 hour\r\n"
            "END:VALARM\r\n"
            "BEGIN:VALARM\r\n"
            "TRIGGER:-PT15M\r\n"
            "ACTION:DISPLAY\r\n"
            "DESCRIPTION:Interview reminder — 15 minutes\r\n"
            "END:VALARM\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )

    # ══════════════════════════════════════════════════════════════════
    #  CONFIRMATION RECORD
    # ══════════════════════════════════════════════════════════════════

    def _create_confirmation(self, invitation: dict, slot: dict) -> dict:
        confirmation_id = str(uuid.uuid4())
        confirmation = {
            "id":              confirmation_id,
            "invitation_id":   invitation["id"],
            "candidate_email": invitation["candidate_email"],
            "candidate_name":  invitation["candidate_name"],
            "recruiter_email": slot["recruiter_email"],
            "job_title":       invitation["job_title"],
            "slot":            slot,
            "confirmed_at":    datetime.now().isoformat(),
            "status":          "confirmed",
        }
        with open(os.path.join(self.confirmations_folder,
                               f"{confirmation_id}.json"), "w") as f:
            json.dump(confirmation, f, indent=2)
        return confirmation

    # ══════════════════════════════════════════════════════════════════
    #  EMAIL SENDER
    # ══════════════════════════════════════════════════════════════════

    def _send_email(self, to_email: str, subject: str, html_content: str,
                    ics_content: str = None, ics_filename: str = "interview.ics"):
        try:
            msg = MIMEMultipart("mixed")
            msg["Subject"] = subject
            msg["From"]    = self.from_email
            msg["To"]      = to_email
            msg.attach(MIMEText(html_content, "html", "utf-8"))

            if ics_content:
                ics_part = MIMEBase("text", "calendar",
                                    method="REQUEST", charset="utf-8")
                ics_part.set_payload(ics_content.encode("utf-8"))
                encoders.encode_base64(ics_part)
                ics_part.add_header("Content-Disposition", "attachment",
                                    filename=ics_filename)
                ics_part.add_header("Content-Type",
                                    "text/calendar; method=REQUEST; charset=utf-8")
                msg.attach(ics_part)

            if self.smtp_username and self.smtp_password:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
                server.quit()
                print(f"Email sent to {to_email}")
            else:
                email_dir = os.path.join(self.config.BASE_DIR, "interviews", "emails")
                os.makedirs(email_dir, exist_ok=True)
                base_name = uuid.uuid4().hex
                with open(os.path.join(email_dir, f"{base_name}.html"),
                          "w", encoding="utf-8") as f:
                    f.write(html_content)
                if ics_content:
                    with open(os.path.join(email_dir, f"{base_name}.ics"),
                              "w", encoding="utf-8") as f:
                        f.write(ics_content)
                print(f"Email saved (SMTP not configured): {base_name}.html")

        except Exception as e:
            print(f"Email error to {to_email}: {e}")

    # ══════════════════════════════════════════════════════════════════
    #  QUERY HELPERS
    # ══════════════════════════════════════════════════════════════════

    def get_available_slots(self, slot_group_id: str) -> list:
        group_path = os.path.join(self.slots_folder, f"group_{slot_group_id}.json")
        if not os.path.exists(group_path):
            return []
        with open(group_path) as f:
            group = json.load(f)
        return [s for s in group["slots"] if s["status"] == "available"]

    def get_scheduled_interviews(self, recruiter_email: str = None) -> list:
        interviews = []
        for file in os.listdir(self.confirmations_folder):
            if not file.endswith(".json"):
                continue
            with open(os.path.join(self.confirmations_folder, file)) as f:
                conf = json.load(f)
            if recruiter_email and conf.get("recruiter_email") != recruiter_email:
                continue
            interviews.append(conf)
        interviews.sort(key=lambda x: x["slot"]["date"] + x["slot"]["start_time"])
        return interviews

    def get_invitation_by_token(self, token: str):
        path = os.path.join(self.invitations_folder, f"{token}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def get_all_slot_groups(self) -> list:
        groups = []
        for file in os.listdir(self.slots_folder):
            if file.startswith("group_") and file.endswith(".json"):
                with open(os.path.join(self.slots_folder, file)) as f:
                    groups.append(json.load(f))
        groups.sort(key=lambda x: x["created_at"], reverse=True)
        return groups

    def extract_email_from_resume(self, parsed_data: dict):
        text    = parsed_data.get("text", "")
        pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails  = re.findall(pattern, text)
        return emails[0] if emails else None

    # Keep book_slot for compatibility with existing routes
    def book_slot(self, token: str, slot_id: str,
                  candidate_email: str, candidate_name: str) -> dict:
        inv_path = os.path.join(self.invitations_folder, f"{token}.json")
        if not os.path.exists(inv_path):
            return {"success": False, "error": "Invalid token"}
        with open(inv_path) as f:
            invitation = json.load(f)
        slot_path = os.path.join(self.slots_folder, f"{slot_id}.json")
        if not os.path.exists(slot_path):
            return {"success": False, "error": "Slot not found"}
        with open(slot_path) as f:
            slot = json.load(f)
        confirmation = self._create_confirmation(invitation, slot)
        return {"success": True, "confirmation": confirmation, "slot": slot}