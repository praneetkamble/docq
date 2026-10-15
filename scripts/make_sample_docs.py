"""Generate the sample PDF corpus used by the demo and the evaluation.

Every document here is FICTIONAL and written by the project author for this repo
(no real company, clinic or product). One PDF page == one list entry below, so page
numbers used as evaluation ground truth are exact.
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Spacer

DOCS = {
"Meridian_Tech_Employee_Handbook": ("Meridian Tech Employee Handbook (fictional)", [
 ("Working Hours and Remote Work", [
  "Core hours at Meridian Tech are 10:30 to 16:30 IST, Monday to Friday. Outside core hours employees may arrange their own schedule with their team lead.",
  "After completing the 90-day probation, employees may work remotely up to three days per week. During probation, remote work is limited to one day per week.",
  "Remote employees must reply on the team chat within 30 minutes during core hours. Remote workers receive an internet allowance of Rs 1,200 per month, paid with salary.",
  "The office is closed on the second Saturday of every month and on the 14 public holidays listed in the annual calendar."]),
 ("Leave Policy", [
  "Full-time employees accrue 24 days of paid leave per year, at two days per month. Up to 10 unused days can be carried forward into the next calendar year; the rest lapse.",
  "Sick leave is 12 days per year. A medical certificate is required when the absence is longer than two consecutive days.",
  "Parental leave is 26 weeks for the primary caregiver and 2 weeks for the secondary caregiver. Bereavement leave is 5 days for the death of an immediate family member.",
  "Leave of more than three days must be requested at least 5 working days in advance through the HR portal."]),
 ("Expenses and Travel", [
  "Expense claims must be submitted within 30 days of the expense, with original receipts attached. Late claims are rejected unless the finance team approves an exception.",
  "Meals during business travel are reimbursed up to Rs 1,500 per day. Flights must be economy class for journeys under 6 hours.",
  "Hotel stays are reimbursed up to Rs 6,500 per night in metro cities and Rs 4,500 per night elsewhere. Any single claim above Rs 25,000 needs approval from a director.",
  "Approved claims are paid with the next payroll cycle, usually by the 7th of the month. Each employee also has a learning budget of Rs 30,000 per year for courses and books."]),
 ("Performance and Conduct", [
  "Performance reviews take place twice a year, in April and October, using a rating scale from 1 to 5. A rating of 4 or higher makes an employee eligible for the annual bonus pool.",
  "The standard notice period is 60 days; during probation it is 30 days.",
  "Harassment complaints go to the Internal Committee, which acknowledges every complaint within 48 hours and concludes its inquiry within 90 days.",
  "Gifts from clients or vendors worth more than Rs 3,000 must be declared to the compliance officer."]),
]),
"Meridian_Tech_Information_Security_Policy": ("Meridian Tech Information Security Policy (fictional)", [
 ("Accounts and Passwords", [
  "Passwords must be at least 14 characters long. Rotation is not required unless a compromise is suspected, because forced rotation leads to weaker passwords.",
  "Multi-factor authentication is mandatory on every company account. Access to production administration also requires a hardware security key.",
  "The company provides the VaultKeep password manager to every employee. Accounts are locked for 15 minutes after 5 failed sign-in attempts. Shared accounts are prohibited."]),
 ("Data Classification and Retention", [
  "Company data has four classification levels: Public, Internal, Confidential and Restricted.",
  "Restricted data, such as customer payment details and health records, must be encrypted with AES-256 at rest and with TLS 1.2 or higher in transit. Restricted data must never be stored on laptops or USB drives.",
  "Confidential data may be emailed only to addresses inside the company domain.",
  "Customer logs are retained for 13 months and then deleted. Encrypted backups are kept for 35 days."]),
 ("Incident Response", [
  "Any suspected security incident must be reported to security@meridiantech.example within 1 hour of discovery. Employees can also use the Report Phishing button in the mail client.",
  "Incidents are rated SEV1 to SEV3. For a SEV1 incident the on-call engineer responds within 15 minutes, 24 hours a day.",
  "A written postmortem is due within 5 working days of closing an incident. If personal data was breached, the regulator is notified within 72 hours."]),
 ("Devices", [
  "All company laptops use full-disk encryption and lock automatically after 5 minutes of inactivity.",
  "Operating system updates must be installed within 14 days of release. Critical security patches must be installed within 48 hours.",
  "Personal phones may access company email only after installing the MDM profile. A lost or stolen device must be reported immediately so it can be remotely wiped."]),
]),
"AquaPure_W200_User_Manual": ("AquaPure W200 Water Purifier User Manual (fictional)", [
 ("Safety and Installation", [
  "Use the AquaPure W200 only with potable water between 5 and 35 degrees Celsius. Do not connect it to hot water lines.",
  "The inlet water pressure must be between 1.0 and 4.0 bar. Install the unit under a sink or on a countertop with at least 15 cm of free space behind it.",
  "The purifier runs on a 24 V DC adapter rated at 40 W. The storage tank holds 8 litres and the purified water flow rate is 2.5 litres per minute."]),
 ("Filter Replacement", [
  "The W200 has three filtration stages. Replace the sediment pre-filter every 6 months, the carbon block every 12 months and the RO membrane every 24 months.",
  "The indicator light turns amber when a filter has reached 90 percent of its life.",
  "After replacing a filter, hold the reset button for 5 seconds until the unit beeps twice. Flush new filters by running water through for 10 minutes before drinking."]),
 ("Troubleshooting", [
  "A blinking red light means a leak was detected. Turn off the water supply and check all fittings.",
  "If no water comes out, check that the inlet valve is open and that the supply pressure is not below 1.0 bar. A bad taste usually means the carbon block is exhausted.",
  "Error code E3 indicates a fault in the TDS sensor. Error code E5 means the tank-full sensor is not responding. To pair the unit with Wi-Fi, hold the Wi-Fi button for 3 seconds."]),
 ("Warranty and Support", [
  "The unit carries a 24-month warranty and filters are covered for 6 months. Register the product within 30 days of purchase to activate the warranty.",
  "The warranty does not cover damage caused by supply water with more than 1500 ppm TDS.",
  "Customer support is available from 9 am to 6 pm, Monday to Saturday, on 1800-000-0142 or at support@aquapure.example."]),
]),
"Sunrise_Family_Clinic_Patient_Guide": ("Sunrise Family Clinic Patient Guide (fictional)", [
 ("Hours and Appointments", [
  "Sunrise Family Clinic is open Monday to Saturday from 8:00 to 20:00. On Sundays only emergency consultations are available, from 10:00 to 14:00.",
  "Appointments can be booked online or by phone. Please cancel at least 4 hours in advance; later cancellations are charged a fee of Rs 300.",
  "Arrive 10 minutes before your slot. New patients should bring a photo ID and any previous medical records. Walk-in tokens are issued from 7:30 each morning."]),
 ("Billing and Insurance", [
  "The consultation fee is Rs 600 for a general physician and Rs 900 for a specialist. A follow-up visit within 7 days of the first visit is free.",
  "Cashless treatment is available with 14 partner insurers. For other insurers the clinic submits the claim form within 3 working days.",
  "Payment can be made by UPI, card or cash, and receipts are sent by email. Patients aged 60 and above get a 10 percent discount on consultation fees."]),
 ("Tests and Vaccinations", [
  "Fasting for 10 to 12 hours is required before blood glucose and lipid profile tests; plain water is allowed.",
  "Blood test reports are ready within 24 hours. Culture reports take 3 days.",
  "The child vaccination clinic runs on Tuesday and Thursday mornings. The seasonal flu vaccine is available from October to January and costs Rs 850. Prescription refill requests need 2 working days."]),
]),
"Greenfield_Apartments_Resident_Handbook": ("Greenfield Apartments Resident Handbook (fictional)", [
 ("Rent and Deposits", [
  "Rent is due on or before the 5th of each month and must be paid by bank transfer; cash is not accepted. After the 5th a late fee of Rs 200 per day applies, capped at Rs 2,000 per month.",
  "The security deposit equals two months of rent. It is refunded within 30 days after move-out, minus the cost of any damage beyond normal wear.",
  "Annual rent increases are capped at 5 percent. Residents must give 60 days notice before vacating."]),
 ("Maintenance", [
  "Routine maintenance requests are raised in the resident app and handled within 3 working days. Plumbing and electrical repairs are included in the rent.",
  "Emergencies such as water leaks, power failure or gas smell should be reported on the 24-hour helpline 022-5550-0199, with a response within 4 hours.",
  "Pest control is done quarterly, the overhead water tanks are cleaned every 6 months, and the lifts are serviced monthly. Repairs caused by misuse of appliances are charged to the resident."]),
 ("Community Rules", [
  "Quiet hours are from 10 pm to 6 am. Each flat may keep up to 2 pets, which must be registered with the office; dogs must be leashed in common areas.",
  "Each flat includes 1 car parking slot; a second slot costs Rs 1,500 per month. Visitor parking is free for 4 hours and then Rs 50 per hour.",
  "The clubhouse can be booked up to 14 days ahead with a refundable deposit of Rs 2,000. Guests staying longer than 15 days must be registered with the office. Smoking is not allowed in common areas."]),
]),
}

def build(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    ss = getSampleStyleSheet()
    for stem, (title, pages) in DOCS.items():
        flow = []
        for i, (heading, paras) in enumerate(pages):
            flow.append(Paragraph(title, ss["Title"]))
            flow.append(Paragraph(f"{i+1}. {heading}", ss["Heading2"]))
            for p in paras:
                flow.append(Paragraph(p, ss["BodyText"]))
                flow.append(Spacer(1, 8))
            if i < len(pages) - 1:
                flow.append(PageBreak())
        SimpleDocTemplate(str(out / f"{stem}.pdf"), pagesize=A4, title=title).build(flow)

if __name__ == "__main__":
    build(Path(__file__).resolve().parent.parent / "sample_docs")
