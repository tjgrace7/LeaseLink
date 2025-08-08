from datetime import datetime
import base64
import tiktoken
from anthropic import Anthropic
import os
import json

def encode_pdf_to_base64(file_path):
    """Encode PDF file to base64 string"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def estimate_total_tokens(prompt, pdf_base64, system_message=""):
    """Estimate total tokens before making API call"""
    encoding = tiktoken.get_encoding('cl100k_base')
    
    prompt_tokens = len(encoding.encode(prompt))
    system_tokens = len(encoding.encode(system_message)) if system_message else 0
    
    # Rough estimate for document tokens (base64 length / 4)
    document_tokens = len(pdf_base64) // 4
    
    total_estimated = prompt_tokens + system_tokens + document_tokens
    
    return {
        'prompt_tokens': prompt_tokens,
        'system_tokens': system_tokens,
        'document_tokens': document_tokens,
        'total_estimated': total_estimated
    }

def claude_extraction(pdf_path, claude_client, max_tokens=500000, verbose=False):
    """
    Extract lease information from PDF using Claude API
    
    Args:
        pdf_path (str): Path to the PDF file
        api_key (str, optional): Anthropic API key. If None, uses ANTHROPIC_API_KEY env var
        max_tokens (int): Maximum tokens allowed (default 180k for safety)
        verbose (bool): Print detailed information
    
    Returns:
        tuple: (extracted_data_json_string, cost_in_dollars)
        Returns (None, 0) if extraction fails
    """
    
    #
    
    
    try:
        # Get current date for context
        now = datetime.now().strftime("%Y/%m/%d")
        start_time = datetime.now()
        
        # Encode the PDF
        pdf_base64 = encode_pdf_to_base64(pdf_path)
        if verbose:
            print("✅ PDF encoded successfully")
        
        # Define the extraction prompt
        prompt = """Read this lease and tell me:

Do calculations as necessary
-lease_execution_date (The Day the lease takes affect (Often handwritten))
-base_rent_monthly (Price of rent per month for the building before expenses. )
-rent_escalation (Give details on the rent schedule during the initial lease. List by date and amount.) 
-security_deposit_amount (The amount the rent has to put as a "down payment" to hold there space. Is paid back at the end of the lease if the property is left in good condition.) 
-base_rent_psf (The Per Square Foot price for Base Rent (Calculate: Annual Rent / SF)) 
-base_rent_annually (Calculate: Base rent amount paid across 12 months) 
-operating_expenses_CAM_psf (CAM + operating expenses per square foot. That includes taxes and Insurance (Calculate: monthly/ SF)
-operating_expenses_CAM_monthly (Monthly estimated amount that tenants are pay in all expenses they are responsible for via the lease. That includes taxes and insurance monthly)  
-property_taxes (A summary of who has responsibility to pay the property taxes for the building.) 
-insurance_costs (A summary of insurance expectations for both the tenant and the Landlord.) 
-CAM_Summary (Make note of any operating expenses not allowed to be charged back to the tenant.) 
-tenant_reimbursements (A summary of the system in which the landlord is able to bill the tenant for expenses they initially paid for or the rights in which the tenants have to recoup the money in which they overpaid for building expenses.) 
-insurance_requirements (Insurance requirements for the renters of the space. (General or more probably liability)) 
-lease_commencement_date (The Day the Lease takes effect, yyyy/mm/dd force into format) 
-lease_expiration_date (Day that the lease ends before any options to renew. Use the formula lease commencement date + term if required. Use yyyy/mm/dd force into format) 
-delivery_possession_date (The day the tenants may access the space, yyyy/mm/dd) 
-CAM_start_date (The date in which the tenant is responsible for paying estimated CAM amounts, yyyy/mm/dd force into format) 
-rent_abatement_end (the date where the tenants rent abatement runs out. Format yyyy/mm/dd) 
-rent_commencement_date (Date that rent starts, yyyy/mm/dd force into format)
-Property_Address (The listed address of the property) 
-suite_identifier (The number or letter of the suite without the address if applicable) 
-lease_term (Length of lease term in months)
-renewal_notice_deadline (The amount of time before the lease expires that the tenant has to let the landlord know they are interested in renewing) 
-option_exercise_deadlines (The time in which the tenant must have accepted the option to renew) 
-renewal_options (The amount of options the tenant has and the terms that change upon the commencement of these options.)
-termination_rights (Any terms that allow either party to terminate the lease early.
-expansion_contraction_rights (The provisions that allow the tenant to grow into more space or shrink out of other space.) 
-co_tenancy_clauses (Obligations that must be met by the landlord in accordance to other tenants and if not met the consequences.) 
-purchase_option (Options the tenant has to purchase the building in within the terms of the lease.) 
-rentable_square_footage (Useable SF + share of common areas (hallways, restrooms, etc.)) 
-usable_square_footage (The amount of square footage in the lease) 
-premises_description (Gives a more general and knowledgeable description of the rentable area) 
-parking_allocation (How much parking the tenant gets.) 
-storage_additional_space (If any storage is allotted or additional space is allotted to the tenant) 
-tenant_maintenance_responsibilities (What is the Lessee's/Tenant responsibility to maintain and repair the unit) 
-landlord_maintenance_responsibilities (What is the Landlord's/Property Managers responsibility to maintain and repair the building/unit) 
-hvac_responsibilities (The HVAC responsibilities in detail) 
-utility_responsibilities (Utility Responsibility in detail) 
-default_and_remedies (The actions and ability to take actions of either part in the event of default by the other.) 
-assignment_and_subletting (Is subletting allowed in the space? If so, under what terms and conditions?) 
-indemnity_clauses (The landlords protection from being held legally liable for anything. (Tenant can't sue landlord)) 
-force_majeure (excuses one or both parties from performing their obligations when extraordinary events occur that are outside their control.) 
-estoppel_certificate_required (The requirement that tenants answer certain questions in certain occasions. Normally when selling or refinancing.) 
-signage_rights (What signage rights the tenant has) 
-permitted_use (What type of business is permitted to use the unit?) 
-exclusive_use_clause (Gives the tenant permission to be the sole operator allowed to do something.) 
-guarantor_information (the details about any person or entity that guarantees the tenant's obligations under the lease.) 
-tenant_improvement_allowance (The amount the Landlord gives to the tenant to improve the property for the tenants use.) 
-holdover_terms (Terms that apply when the tenant overstays their lease without a renewal.) 
-landlord_work (Work that is the responsibility of the landlord, normally before the tenant moves in.) 
-Tenant_work (The work or improvements that the tenant is held responsible if applicable) 
-security_deposit_term (The terms that define security deposit rules.) 
-ROFR_ROFO_clauses (Right of First Refusal clauses or Right of First Offer clauses) 
-security_access_rights (The rights of security and the limits to the landlords access.) 
-exclusivity_rights (Blocks the landlord from allowing any competing business.)

Please respond with a valid JSON object only."""

        system_message = f"""You are a leasing document analyzer. Respond only with a JSON object containing all the requested fields. If information is not available in the document, omit that field. Perform calculations as needed and format dates as yyyy/mm/dd. For Current or Base Rent, CAM etc. 
When extracting cost-related fields (such as rent, CAM charges, or other expenses), use the current date: {now} to determine relevance. example ie if base_monthly_rent starts at $1679 but from 2/1/24-1/31/2025 it should 1782. And we are within that date range use that. If that is the last date range available, because the lease expired or another reason, use that"""

        # Estimate tokens before API call
        token_info = estimate_total_tokens(prompt, pdf_base64, system_message)
        
        if verbose:
            print(f"📊 Token estimation:")
            print(f"  Prompt: {token_info['prompt_tokens']:,}")
            print(f"  System: {token_info['system_tokens']:,}")
            print(f"  Document: {token_info['document_tokens']:,}")
            print(f"  Total: {token_info['total_estimated']:,}")
        
        # Check token limit
        if token_info['total_estimated'] > max_tokens:
            if verbose:
                print(f"❌ Token limit exceeded: {token_info['total_estimated']:,} > {max_tokens:,}")
            return None, 0
        
        # Make API call
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            temperature=0,
            system=system_message,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_base64
                            }
                        }
                    ]
                }
            ],
        )
        
        # Calculate cost
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = (input_tokens * 0.003 / 1000) + (output_tokens * 0.015 / 1000)
        
        # Get extracted data
        extracted_data = response.content[0].text
        
        try: 
            extracted_dict = json.loads(extracted_data)
        except Exception as e:
            print("JSON parsing error in claude response", e)
            raise
        if verbose:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            print(f"✅ Extraction successful!")
            print(f"  Input tokens: {input_tokens:,}")
            print(f"  Output tokens: {output_tokens:,}")
            print(f"  Cost: ${cost:.4f}")
            print(f"  Duration: {duration:.2f}s")
        
        return extracted_dict, cost
        
    except Exception as e:
        if verbose:
            print(f"❌ Error during extraction: {str(e)}")
        return None, 0

