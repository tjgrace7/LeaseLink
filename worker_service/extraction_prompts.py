
import os

prompts = {
    'lease_signed_date': {
        'prompt': """Task:
Extract the date the original lease was executed (signed) by all required parties, as stated in the signature block or execution clause of the ORIGINAL LEASE ONLY.

Rules:
- DO NOT infer from commencement date, effective date, possession date, or rent start date.
- DO NOT return dates from amendments, renewals, extensions, or restatements.
- If the text is an amendment or renewal, IGNORE it completely.
- If multiple signature dates exist, return the LATEST date on which all required parties executed the original lease.
- If no valid original execution date is found, return null. - Return in Format provided in System Prompt

""",
    'required_document(s)': 'Original_Lease',
    'need_current_time': False,
    'minimum_required_confidence': .80},

    
    'latest_lease_modification_signed_date': {

        'prompt': """Task: Extract the execution date of the most recent document that modifies, extends, renews, or otherwise affects the lease terms or economics, 
    including amendments, extensions, renewal letters, or similar agreements. Do not include documents that do not alter lease terms (e.g., estoppels, SNDA). 
    If no such document exists, leave null.
    
    Rules:
    - DO NOT use Original lease signed date
    - Return null if there are no amendements, renewals, or other modification document
    """,
        'required_document(s)': 'Current_Lease',
        'need_current_time': False,
        'minimum_required_confidence': .8},
    'base_rent_amount_current': {
        'prompt': """Task: Extract the Base Rent amount (today's rent) payable for a single billing period reflecting the most recent amendment or escalation in effect.
          
          Rules:

            STEP 1: Identify which period includes TODAY
            STEP 2: Extract ONLY that amount as "value"
            STEP 3: Extract ALL future periods as "future_value" seperate future rents out so they are json parsable
            STEP 4: Extract Dates future rents take effect
            STEP 5: Determine if the rent in the lease is monthly, quarterly, semi-annual, or annual.
            STEP 6: If the rent is not monthly, calculate the proper monthly rent by dividing the rent amount by the number of months in the billing period. EX: Quarterly rent of $3000 = Monthly rent of $1000, Annual rent of $12000 = Monthly rent of $1000 etc.
            STEP 7: Do not Just Return whatever value is in the lease! If the rent is not monthly, you must calculate the monthly rent amount.
            STEP 8: If you are unsure of your calculation, require manual review by setting manual_review to true.
            STEP 9: In the Json Extraction Add a field about CPI_lease. Determine if the rent is a CPI lease or a normal rent lease (cpi_lease: true or false)
            STEP 10: If cpi_lease = true, needs_correction = true

            ⚠️ DO NOT:
            - Choose the highest or most recent amount
            - Include expired periods
            - Include future rent in 'value'""",
        'required_document(s)': 'Current_Lease',
        'need_current_time': True,
        'minimum_required_confidence': .80
    },
    'base_rent_frequency': {
        'prompt': """Task: Identify the billing frequency at which Base Rent is payable under the lease. 
        Rules:
        - Select a value only if explicitly stated. 
        - Do not infer frequency from payment examples or industry norms. 
        - Use other if rent is charged on a non-standard schedule, and unknown if frequency is not clearly stated. 
        - Only use the following values: monthly, quarterly, semi_annual, annual, other, unknown.""",
        'required_document(s)': 'All',
        'need_current_time': False,
        'minimum_required_confidence': .8
    },
    'base_rent_payment_timing': {
        'prompt': """Task: Identify whether Base Rent is payable in advance or in arrears. 
        Rules:
        -Return only if explicitly stated. 
        -Typical language includes “payable in advance on the first day of each month” or “payable in arrears.” 
        -If not clearly stated, select unknown. 
        -Only state the following options: in advance, in arrears, unknown.""",
        'required_document(s)': 'All',
        'need_current_time': False,
        'minimum_required_confidence': .8
    },
    'base_rent_due_day': {
        'prompt': """Task: Extract the calendar day of the month on which Base Rent is due (1–31), if explicitly stated. 
        Rules:
        - Do not infer from examples or customary practice (e.g., “monthly rent” does not imply the 1st). 
        - Leave null if not stated or if rent is not charged monthly.
        - Only return the day (1-31) that the rent is actually due monthly""",
        'required_document(s)': 'All',
        'need_current_time': False,
        'minimum_required_confidence': .8
    },
    'base_rent_effective_date': {
        'prompt': """Task: Extract the effective date on which the current Base Rent amount becomes applicable, such as the start of a new rent step, escalation period, or amendment. 
        Rules:
        - This date may differ from lease commencement or rent commencement. 
        - Do not infer from context.
        - If the current rent is effective from lease commencement and no separate date is stated, leave null.""",
        'required_document(s)': 'Original_Lease',
        'need_current_time': False,
        'minimum_required_confidence': .8
    },
    'base_rent_schedule': {
        'prompt': """Task: Extract the schedule of Base Rent changes over time, including step rents, escalations, and amended rates, when explicitly stated in tabular or clearly defined form. 
        Rules:
         -Each entry should include a start date (or period identifier) and rent amount. 
         - Do not infer missing periods or compute escalations. 
         - If a clear schedule is not stated, leave null. EX: 1/1/2025 - 1/1/2026 Rent $25,000 per month. EX: Month 1-12 Rent $1300, Month 12-24 Rent $1500. 
         - Often in Tables
         - Assume the leases are using mm/dd/yyyy format
         - Return in yyyy/mm/dd format
         - If the leases say Month 1-12 or don't have specific days, Return Periods as stated in lease unless there is a manually entered lease commencement date)
         - If there is a manually added lease commencement date and the dates (Months 1-12) are not present. Calculate the actual dates of the rent schedule
         - If multiple rent schedules, return schedule with most recent dates
         - If All Rent Schedules are expired from current date return null
         """,
         'required_document(s)': 'Current_Lease',
         'need_current_time': True
    },
    'security_type': {
        'prompt': """Task: Identify the form of security required under the lease, such as a cash security deposit, letter of credit, guaranty, or other stated form. 
        Rules:
         - Select only one of the following value based on explicit lease language: cash, letter of credit, guaranty only, other, not stated.
         - Don't Infer From the Lease. Leave Null if Not Stated
         """,
        'required_document(s)': 'All',
        'need_current_time': False
    },
    'security_deposit_amount': {

        'prompt': """Task: Extract the stated dollar amount of a security deposit required under the lease, whether in the form of cash or letter of credit. 
        Rules:
        - Do not include replenishment requirements, reductions over time, or conditional adjustments. 
        - If the security amount is variable or not stated as a fixed dollar amount, leave null.""",
        'required_document(s)': 'Original_Lease',
        'need_current_time': False
    },
    'additional_rent_components': {
        'prompt': """Task: Structured indicators of which expense categories are recoverable under the lease (e.g., CAM, taxes, insurance), and whether recovery is direct or indirect.
            Rules:
             - Describe what is recoverable from the lease
             - Do NOT infer from context
             - Return null if not found""",
        'required_document(s)': 'All',
        'need_current_time': False,
    },
    'additional_rent_billing_method': {
        'prompt': """Task: Determine how Additional Rent is billed under the lease. 
        Rules:
        - If Additional Rent obligations are stated in the lease, select exactly one enum value based only on explicit lease language. 
        - Do not infer based on lease type (e.g., “NNN”). 
        - Choose one of these options: monthly estimated reconciliation, quarterly estimated reconciliation, annual reconciliation only, fixed additional rent, direct pay only, mixed, not stated.
        - If multiple options match pick in order of listed options above""",
        'required_document(s)': 'All',
        'need_current_time': False
    },
    'additional_rent_commencement_date': {
        'prompt': """Task: Extract the date on which the tenant’s obligation to pay Additional Rent (e.g., CAM, taxes, insurance) begins,
         Rules:
         - only if the lease explicitly states a commencement date for Additional Rent that is independent from rent or lease commencement. 
         - Do not infer from rent commencement, possession, or lease commencement. 
         - If no specific date is stated, leave this field null.""",
         'required_document(s)': 'All',
         'need_current_time': False
    },
    'additional_rent_limitations': {
        'prompt': """Task: Extract only limitations, exclusions, caps, expense stops, or constraints on Additional Rent that are explicitly stated in the lease, including (when expressly stated) expense caps, base year definitions,
          excluded cost categories, limits on management or administrative fees, capital expense treatment, expense stops, or audit rights. 
        Rules: 
         - Do not infer or assume exclusions.""",
        'required_document(s)': "All",
        'need_current_time': False,
        'minimum_required_confidence': .7
    },
    'lease_commencement_date': {
        'prompt': """Task: Extract the date of the original lease term begins (“Commencement Date” or “Lease Commencement Date”) as explicitly defined in the lease. 
        Rules: 
        - Return in YYYY-MM-DD. 
        - Do not use execution/effective dates unless the lease explicitly defines them as the commencement date. 
        - Do not infer from possession or rent commencement.
        - If commencement is conditional and no calendar date is stated, return null.
        - If null or No Commencement Date return manual_review = true
        - If the Current Value was changed manually Do Not Adjust""",
        'required_document(s)': 'Original_Lease',
        'need_current_time': False
    },
    'possession_date': {
        'prompt': """Task: Extract the date on which the tenant is delivered or accepts possession of the premises, if explicitly stated. This may be referred to as “Possession Date,” “Delivery Date,” or similar.
         Rules:
         - Do not infer possession from lease commencement, rent commencement, or execution date.
         - If possession is conditional or described relative to another event without a calendar date, leave this field null.""",
         'required_document(s)': 'Original_Lease',
         'need_current_time': False
    },
    'rent_commencement_date': {
        'prompt': """Task: Extract the date on which Base Rent begins to accrue, if explicitly stated. This may occur after possession, after a free rent period, or upon satisfaction of conditions. 
        Rules: 
        - Do not infer from lease commencement or possession date. 
        - If rent commencement is described only relative to another event without a fixed date, leave this field null. 
        - Only enter a date or leave null.""",
        'required_document(s)': 'All',
        'need_current_time': False
    },
    'rent_abatement_end_date': {
        'prompt': """Task: Extract the last calendar date on which Base Rent abatement or free rent applies, if explicitly stated. This represents the end of the abatement period, not the start. 
        Rules:
        - If abatement is described only as a duration (e.g., “first three months”) or relative to another event without a stated end date, leave this field null.""",
        'required_document(s)': 'All',
        'need_current_time': False
    },
    'lease_expiration_date': {
        'prompt': """Task: Extract the stated expiration date of the lease term, excluding any unexercised renewal or extension options. 
        Rules:
        - Do not extend the expiration date based on options unless the lease explicitly states that the term includes such periods. 
        - If expiration is defined only by term length without a calendar date, leave this field null.
        - If the lease states a term length (e.g., "five (5) years") and an effective date (e.g., "shall take effect as of August 1, 2021"):
        - Calculate the end date as: end_date = effective_date + term_years − 1 day.     
        - Example: effective date 2021-08-01 with a 5-year term ends on 2026-07-31. 
        - If today is on or after the effective/start date AND on or before the calculated end date, the lease is active. """,
        'required_document(s)': 'All',
        'need_current_time': False
    },
    'lease_term_months': {
        'prompt': """Task: Extract the stated length of the lease term in months, if explicitly defined. 
        Rules:
        - If the lease states the term in years, convert to months. 
        - Do not infer the term length from commencement and expiration dates unless both are explicitly stated and clearly define the full term. 
        - If the term length is conditional or unclear, leave this field null.""",
        'required_document(s)': 'Current_Lease',
        'need_current_time': False
    },
    'rights_index': {
        'prompt': """Task: "Identify whether the lease expressly grants or expressly excludes any of the following tenant rights or option clauses. 
         Rights to Check:
            - renewal_option
            - expansion_rights
            - termination_option
            - contraction_rights
            - rofr
            - rofo
            - purchase_option
            - co_tenancy
            - assignement_subletting_restrictions
         Rules:
          - For each right, include a key only if the lease explicitly states the right is granted or explicitly states it is excluded. 
          - Do not infer rights based on lease type, market norms, or silence.
          - If the lease is silent or unclear regarding a right, omit the key entirely.
          - This index reflects presence only and does not require summarizing terms, conditions, timing, or economics.
          - Set the value to true if the right is expressly granted.
          - Set the value to false only if the lease expressly states the right does not apply or is waived.
          - Use the exact right term in the list above. Do not alter in any way. Say true or false for each""",
          'required_document(s)': 'All',
          'need_current_time': False
    },
    'renewal_options_summary': {
        'prompt': """Task: Summarize the tenant’s future or upcoming renewal or extension rights, if any, including the number of options, option term lengths, and any stated rent determination method. 
        Rules:
        - Do not calculate future rent or interpret market terms. 
        - If the lease does not state any renewal or extension rights, return null. 
        - If renewal or extension rights are stated but such options are already expired or in the past, return null. 
        - Do not state ‘none’ or ‘not applicable’; leave null if not stated.
        - Use Current Time to determine future options. 
        - Exclude Options where the dealine already passed""",
        'required_document(s)': 'Current_Lease',
        'need_current_time': True
    },
    'renewal_notice_requirements_summary': {
        'prompt': """Task: Summarize the notice requirements for exercising renewal or extension options, including timing windows, delivery method, and any stated conditions. 
        Rules: 
        - Do not calculate dates or deadlines. 
        - If notice requirements are not clearly stated, leave null. 
        - Do not state ‘none’ or ‘not applicable’; leave null if not stated. 
        - If their are no renewal options or rights left in the lease leave null.""",
        'required_document(s)': 'Current_Lease',
        'need_current_time': True
    },
    'premises_description': {
        'prompt': """Task: Summarize how the leased premises are defined or described, including suite/unit identifier(s), building or address reference if stated, and the type of area measurement used (e.g., RSF/USF) only if expressly stated.
Rules:
- Include any expressly stated inclusions or exclusions that materially define the premises (e.g., including mezzanine, excluding exterior patio or storage).
- If the premises cannot be clearly identified from the provided documents, leave null.""",
        'required_document(s)': "Original_Lease",
        'need_current_time': False
    },
    'parking_allocation': {
        'prompt': """Task: Summarize the tenant’s parking rights, if any, including number of spaces, type (reserved or unreserved), location (garage, lot, or area) if specified, and any charges or validation obligations if stated. 
        Rules: 
        - Do not infer parking ratios or rights from zoning, codes, or building standards. 
        - If parking is common area with no tenant-specific rights, summarize that briefly.
        - If parking is not addressed, leave null.""",
        'required_document(s)': 'All',
        'need_current_time': False,
        'minimum_required_confidence': .7
    },
    'tenant_maintenance_responsibilities': {
        'prompt': """Task: Summarize the Tenant’s maintenance, repair, and replacement obligations, including any expressly retained responsibilities. Include any stated limitations or conditions.
        Rules:
         - Most Leases will have Tenant Responsibilities
         - Return any Repairs, Maintenance, or types of responsibility the tenant is obligated to fulfill
         - The responsible party is the party who is financially responsible. EX: if the landlord fixes a water fountain, but the tenant is obligated to fix, that is tenant responsibility or vice-versa""",
         'required_document(s)': 'All',
         'need_current_time': False
    },
    'landlord_maintenance_responsibilities': {
        'prompt': """Task: Summarize the landlord’s maintenance, repair, and replacement obligations, including any expressly retained responsibilities. Include any stated limitations or conditions. 
        Rules:
        - Most Leases will have Landlord Responsibilities
        - Return any Repairs, Maintenance, or types or responsibility the tenant is obligated to fulfill
        - The responsible party is the party who is financially responsible. EX: if the landlord fixes a water fountain, but the tenant is obligated to fix, that is tenant responsibility or vice-versa""",
        'required_document(s)': 'All',
        'need_current_time': False
    },
    'hvac_responsibilities': {
        'prompt': """Task: Summarize responsibility for HVAC maintenance, repair, and replacement, including any stated distinctions between maintenance versus replacement, unit types or service areas, cost-sharing thresholds, 
        warranties, service-contract requirements, or inspection obligations. 
        Rules:
        - If HVAC is addressed only generally under maintenance clauses, summarize only what is explicitly attributable to HVAC. 
        - If HVAC is not addressed, leave null.""",
        'required_document(s)': 'All',
        'need_current_time': False
    },
    'utility_responsibilities': {
        'prompt': """Task: Summarize which utilities the tenant is responsible to pay and how they are charged (direct meter, submeter, or landlord billing), only if explicitly stated. 
        Rules:
        - Include any utilities expressly provided by the landlord. 
        - If utilities are not addressed, leave null.""",
        'required_document(s)': 'All',
        'need_current_time': False
    },
    'permitted_use': {
        'prompt': """Task: Extract only the clause that expressly states what business activity the tenant is authorized to conduct within the premises. Summarize the permitted use in clear, concise language, limited strictly to the specific business activities described in the lease.

        Rules:
        - Do not infer use from tenant name, industry, branding, or context.
        - Extract only affirmative language granting the tenant the right to operate a specific business or use.
        - Ignore compliance with laws, zoning ordinances, governmental regulations, or “lawful use.”
        - If no affirmative permitted use clause exists, return null.""",
        'required_document(s)': 'All',
        'need_current_time': False
    },


}   