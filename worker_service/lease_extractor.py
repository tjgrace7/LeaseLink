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
- lease_expiration_date
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
- rent_abatement_end (the date where the tenants rent abatement runs out.)
- rent_commencement_date
- renewal_notice_deadline
- CAM_start_date
- option_exercise_deadlines (The time in which the tenant must have accepted the option to renew)
- delivery_possession_date (The day the tenants may access the space)
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
- lease_expiration_date
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
- rent_abatement_end (the date where the tenants rent abatement runs out.)
- rent_commencement_date
- renewal_notice_deadline
- CAM_start_date
- option_exercise_deadlines (The time in which the tenant must have accepted the option to renew)
- delivery_possession_date (The day the tenants may access the space)
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
        raise e