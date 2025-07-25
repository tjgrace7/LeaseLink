from qdrant_client.http.models import  Filter, FieldCondition, MatchValue
import json
from dotenv import load_dotenv
import tiktoken

from memory_profiler import profile

@profile
def get_relevant_chunks_from_lease(collection_Name, q_client, chatGPT, session_id, top_k=30) -> dict:
    try:
        #ChatGPT analysis lease to determine lease type, effective, and execution dates prompt below
        query = "Classify this lease and extract key details like term, rent, maintenance, taxes, rent increases, maintenance terms, insurance, CAMS, square-footage, state-of-registration, mailing address, effective date, and execution date"
        print("Get_relevant_chunk_from_lease_inner_function")
        query_vector = ''
        try:

            prompt_embed = chatGPT.embeddings.create(
                input=query,
                model="text-embedding-3-large"
            )
            encoding = tiktoken.encoding_for_model("text-embedding-3-large")
            embedding_token_count = len(encoding.encode(query))
            embeddingcost = embedding_token_count*.00000013
            query_vector = prompt_embed.data[0].embedding
        except Exception as e:
            print("Error Embedding file", e)
        results = []
        try:
            results = q_client.search(
                collection_name = collection_Name,
                query_vector = query_vector,
                limit = top_k,
                with_payload=True,
                with_vectors=False,
                query_filter=Filter(
                    must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
                )
            )
            if not results:
                raise ValueError(f"No Chunks found for session_id: {session_id}")
        except Exception as e:
            print("Error Getting Response from Qdrant")
        context = "\n\n".join([r.payload.get("text", "") for r in results])
        prompt = f"""
Here is the lease text:
{context}

---

**Task 1:** Classify the document into one of the following types:
- "main_lease"
- "amendment"
- "exhibit"
- "guaranties"
- "assignments"
- "addendum"
- "sublease_agreements"
- "notice_documents"
- "letters_of_work"
- "other"

**Task 2:** Extract the following fields if available:
- effective_date (Must be in yyyy/mm/dd. Omit if not)
- execution_date (Must be in yyyy/mm/dd. Omit if not)
- term
- current_rent
- rent_increase
- maintenance_terms
- taxes
- insurance
- CAMS
- square_footage
- state_of_registration
- mailing_address
- details
- lease_execution_date (the Day the lease was signed, Must be in yyyy/mm/dd. Omit if not)
- lease_commencement_date (The Day the Lease takes effect, Must be in yyyy/mm/dd. Omit if not)
- Property_Address
- suite_identifier (The number or letter of the suite without the address)
- lease_type (NNN, Gross, Percentage)
- lease_expiration_date (Must be in yyyy/mm/dd. Omit if not)
- lease_term
- base_rent_monthly
- rent_escalation
- security_deposit_amount
- base_rent_psf (The Per Square Foot Base Rent (Annual Rent / SF))
- base_rent_annually
- operating_expenses_CAM_psf (CAM expenses per square foot)
- operating_expenses_CAM_monthly
- CAM_Summary
- property_taxes
- insurance_costs
- tenant_reimbursements
- rent_abatement_end (the date where the tenants rent abatement runs out, Must be in yyyy/mm/dd. Omit if not)
- rent_commencement_date (Must be in yyyy/mm/dd. Omit if not)
- renewal_notice_deadline 
- CAM_start_date (Must be in yyyy/mm/dd. Omit if not)
- option_exercise_deadlines (The time in which the tenant must have accepted the option to renew)
- delivery_possession_date (The day the tenants may access the space, Must be in yyyy/mm/dd. Omit if not)
- renewal_options
- termination_rights
- expansion_contraction_rights (The provisions that allow the tenant to grow into more space or shrink out of other space.)
- co_tenancy_clauses
- purchase_option
- rentable_square_footage (Useable SF + share of common areas (hallways, restrooms, etc.))
- usable_square_footage (The amount of square footage the tenant occupies)
- premises_description
- parking_allocation
- storage_additional_space
- tenant_maintenance_responsibilities
- landlord_maintenance_responsibilities
- hvac_responsibilities
- utility_responsibilities
- default_and_remedies (The actions and ability to take actions of either part in the event of default by the other.)
- assignment_and_subletting (What is permissible by the tenant if they desire to assign the lease or sublet the space.)
- insurance_requirements
- indemnity_clauses
- compliance_with_laws
- force_majeure
- estoppel_certificate_required (Details)
- signage_rights
- permitted_use
- exclusive_use_clause
- guarantor_information
- tenant_improvement_allowance
- holdover_terms (Terms that apply when the tenant overstays their lease without a renewal.)
- landlord_work
- Tenant_work
- security_deposit_term
- ROFR_ROFO_clauses (Right of First Refusal clauses or Right of First Offer clauses)
- security_access_rights
- exclusivity_rights



Respond only with a JSON object. Do not add null values. Omit missing fields. Do not include any text outside the JSON object. **Do not add fields that may apply. Only send keys that are listed above. Errors will occur if extra fields send** Dates must be formatted as yyyy/mm/dd. Dates not in this format will fail (Omit if not complete)
Items in () describe the item being searched for do not include in json response,

DO NOT CHANGE THE TITLE OF ANY FIELDS

If json key is N/A just exclude
"""
        print("prompt")
        chat_response = chatGPT.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a leasing document analyzer. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature = 0.2
        )
        try:
            json_start=chat_response.choices[0].message.content.find("{")
            token_usage = chat_response.usage
            prompt_tokens = token_usage.prompt_tokens
            prompt_cost = (prompt_tokens/1000*.01) + (embedding_token_count/1000*.00013)
            completion_tokens = token_usage.completion_tokens
            completion_cost = completion_tokens/1000*.03
            print("json_start:", json_start)
            total_cost = prompt_cost + completion_cost + embeddingcost
            json_string=chat_response.choices[0].message.content[json_start:]
            print("json_string:", json_string)
            del prompt_tokens, prompt_cost, completion_tokens, completion_cost, prompt, context, results, query_vector,  
            return json.loads(json_string), total_cost
        except Exception as e:
            print("Failed to parse JSON:", chat_response.choices[0].message.content)
            raise e
    except Exception as e:
        print("Error Parsing Lease", e)
        raise e@profile
def get_relevant_chunks_from_lease(collection_Name, q_client, chatGPT, session_id, top_k=30) -> dict:
    try:
        #ChatGPT analysis lease to determine lease type, effective, and execution dates prompt below
        query = "Classify this lease and extract key details like term, rent, maintenance, taxes, rent increases, maintenance terms, insurance, CAMS, square-footage, state-of-registration, mailing address, effective date, and execution date"
        print("Get_relevant_chunk_from_lease_inner_function")
        query_vector = ''
        try:

            prompt_embed = chatGPT.embeddings.create(
                input=query,
                model="text-embedding-3-large"
            )
            encoding = tiktoken.encoding_for_model("text-embedding-3-large")
            embedding_token_count = len(encoding.encode(query))
            embeddingcost = embedding_token_count*.00000013
            query_vector = prompt_embed.data[0].embedding
        except Exception as e:
            print("Error Embedding file", e)
        results = []
        try:
            results = q_client.search(
                collection_name = collection_Name,
                query_vector = query_vector,
                limit = top_k,
                with_payload=True,
                with_vectors=False,
                query_filter=Filter(
                    must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
                )
            )
            if not results:
                raise ValueError(f"No Chunks found for session_id: {session_id}")
        except Exception as e:
            print("Error Getting Response from Qdrant")
        context = "\n\n".join([r.payload.get("text", "") for r in results])
        prompt = f"""
Here is the lease text:
{context}

---

**Task 1:** Classify the document into one of the following types:
- "main_lease"
- "amendment"
- "exhibit"
- "guaranties"
- "assignments"
- "addendum"
- "sublease_agreements"
- "notice_documents"
- "letters_of_work"
- "other"

**Task 2:** Extract the following fields if available:
- effective_date
- execution_date
- term
- current_rent
- rent_increase
- maintenance_terms
- taxes
- insurance
- CAMS
- square_footage
- state_of_registration
- mailing_address
- details
- lease_execution_date (the Day the lease was signed)
- lease_commencement_date (The Day the Lease takes effect)
- Property_Address
- suite_identifier (The number or letter of the suite without the address)
- lease_type (NNN, Gross, Percentage)
- lease_expiration_date (Day that the lease ends before any options to renew.)
- lease_term (Length of lease in years/ months)
- base_rent_monthly (Amount of rent for the building before expenses)
- rent_escalation (The rent increase within the current term of the lease)
- security_deposit_amount (The amount the rent has to put as a "down payment" to hold there space. Is paid back at the end of the lease if the property is left in good condition.)
- base_rent_psf (The Per Square Foot Base Rent (Annual Rent / SF))
- base_rent_annually (Base rent amount paid across 12 months)
- operating_expenses_CAM_psf (CAM expenses per square foot)
- operating_expenses_CAM_monthly (Monthly estimated amount that tenants are pay in all expenses they are responsible for via the lease)
- CAM_Summary (A summary of the Common Area Maintenance and who is responsible for expenses.)
- property_taxes (A summary of who has responsibility to pay the property taxes for the building.)
- insurance_costs (A summary of insurance expectations for both the tenant and the Landlord.)
- tenant_reimbursements (A summary of the system in which the landlord is able to bill the tenant for expenses they initially paid for or the rights in which the tenants have to recoup the money in which they overpaid for building expenses.)
- rent_abatement_end (the date where the tenants rent abatement runs out.)
- rent_commencement_date (Date that rent starts)
- renewal_notice_deadline (The amount of time before the lease expires that the tenant has to let the landlord know they are interested in renewing)
- CAM_start_date (The date in which the tenant is responsible for paying estimated CAM amounts)
- option_exercise_deadlines (The time in which the tenant must have accepted the option to renew)
- delivery_possession_date (The day the tenants may access the space)
- renewal_options (The amount of options the tenant has and the terms that change upon the commencement of these options.)
- termination_rights (Any terms that allow either party to terminate the lease early.)
- expansion_contraction_rights (The provisions that allow the tenant to grow into more space or shrink out of other space.)
- co_tenancy_clauses (Obligations that must be met by the landlord in accordance to other tenants and if not met the consequences.)
- purchase_option (Options the tenant has to purchase the building in within the terms of the lease.)
- rentable_square_footage (Useable SF + share of common areas (hallways, restrooms, etc.))
- usable_square_footage (The amount of square footage the tenant occupies)
- premises_description (Gives a more general and knowledgeable description of the rentable area)
- parking_allocation (How much parking the tenant gets.)
- storage_additional_space (If any storage is allotted or additional space is allotted to the tenant)
- tenant_maintenance_responsibilities (Tenant's maintenance responsibilities.)
- landlord_maintenance_responsibilities (Landlord's Maintenance Responsibilities)
- hvac_responsibilities (The HVAC responsibilities in detail)
- utility_responsibilities (Utility Responsibility in detail)
- default_and_remedies (The actions and ability to take actions of either part in the event of default by the other.)
- assignment_and_subletting (What is permissible by the tenant if they desire to assign the lease or sublet the space.)
- insurance_requirements (Insurance requirements for the renters of the space. (General or more probably liability))
- indemnity_clauses (The landlords protection from being held legally liable for anything. (Tenant can't sue landlord))
- compliance_with_laws (Tenants responsibility to be compliant with laws, federal and state.)
- force_majeure (excuses one or both parties from performing their obligations when extraordinary events occur that are outside their control.)
- estoppel_certificate_required (The requirement that tenants answer certain questions in certain occasions. Normally when selling  or refinancing.)
- signage_rights (What signage rights the tenant has)
- permitted_use (What the tenant is allowed to use the space for.)
- exclusive_use_clause (Gives the tenant permission to be the sole operator allowed to do something.)
- guarantor_information (the details about any person or entity that guarantees the tenant’s obligations under the lease.)
- tenant_improvement_allowance (The amount the Landlord gives to the tenant to improve the property for the tenants use.)
- holdover_terms (Terms that apply when the tenant overstays their lease without a renewal.)
- landlord_work (Work that is the responsibility of the landlord, normally before the tenant moves in.)
- Tenant_work (The work or improvements that the tenant is held responsible for upon receiving access to the space)
- security_deposit_term (The terms that define security deposit rules.)
- ROFR_ROFO_clauses (Right of First Refusal clauses or Right of First Offer clauses)
- security_access_rights (The rights of security and the limits to the landlords access.)
- exclusivity_rights (Blocks the landlord from allowing any competing business.)



Respond only with a JSON object. Do not add null values. Omit missing fields. Do not include any text outside the JSON object. **Do not add fields that may apply. Only send keys that are listed above. Errors will occur if extra fields send** Dates must be formatted as yyyy/mm/dd. Dates not in this format will fail (Omit if not complete)
Items in () describe the item being searched for do not include in json response,

DO NOT CHANGE THE TITLE OF ANY FIELDS

Return ONLY a valid JSON object. 
DO NOT include any additional explanation, commentary, or fields. 
DO NOT include null values. 
Any output that violates this will be rejected.
"""
        print("prompt")
        chat_response = chatGPT.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a leasing document analyzer. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature = 0
        )
        try:
            json_start=chat_response.choices[0].message.content.find("{")
            token_usage = chat_response.usage
            prompt_tokens = token_usage.prompt_tokens
            prompt_cost = (prompt_tokens/1000*.01) + (embedding_token_count/1000*.00013)
            completion_tokens = token_usage.completion_tokens
            completion_cost = completion_tokens/1000*.03
            print("json_start:", json_start)
            total_cost = prompt_cost + completion_cost + embeddingcost
            json_string=chat_response.choices[0].message.content[json_start:]
            print("json_string:", json_string)
            del prompt_tokens, prompt_cost, completion_tokens, completion_cost, prompt, context, results, query_vector,  
            return json.loads(json_string), total_cost
        except Exception as e:
            print("Failed to parse JSON:", chat_response.choices[0].message.content)
            raise e
    except Exception as e:
        print("Error Parsing Lease", e)
        raise e