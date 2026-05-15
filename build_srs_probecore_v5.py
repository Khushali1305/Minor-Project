"""
build_srs_probecore_v5.py
=========================
Steps 2-5 of SRS-ProbeCore v5 build pipeline.

USAGE
-----
    python3 build_srs_probecore_v5.py

INPUT  : srs_probecore_v4_probed_final.json   (your updated file with Cat-D dropped)
OUTPUTS: SRS-ProbeCore-v5-frozen.json          (main benchmark, frozen)
         nfr_shortcut_minibench_v1.json        (standalone NFR supplementary benchmark)

WHAT THIS SCRIPT DOES
---------------------
Step 2 — Injects one hand-crafted invariance probe (Strategy 1: structural paraphrase)
         for each Category A and B item that currently has only 2 probes.
         Category C items (epistemic/sensitive) are left untouched.

Step 3 — Populates generation_method field for all directional and shortcut probes
         that currently have generation_method=None.
         directional -> 'rule_based_modal_substitution'
         shortcut    -> 'rule_based_distractor_insertion'

Step 4 — Saves the NFR mini-benchmark as a separate standalone JSON file.
         18 probes across 8 NFR categories. NOT injected into main benchmark.

Step 5 — Runs sanity checks, prints final stats, saves frozen v5 JSON.

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
- It does not modify the input file.
- It does not auto-execute without you reviewing the dry-run output first.
- Set DRY_RUN = True to see what would change before writing anything.
"""

import json
import copy
from collections import Counter

# =============================================================================
# CONFIGURATION — edit these paths if needed
# =============================================================================

INPUT_PATH   = '/teamspace/studios/this_studio/parse/srs_probecore_v4_probed_final.json'
OUTPUT_MAIN  = '/teamspace/studios/this_studio/parse/SRS-ProbeCore-v5-frozen.json'
OUTPUT_NFR   = '/teamspace/studios/this_studio/parse/nfr_shortcut_minibench_v1.json'

# Set to True to see what would change without writing any file
DRY_RUN = False

# =============================================================================
# CATEGORY C ITEM IDs — these 12 items are left at 2-probe intentionally
# (epistemic/sensitive modals — Strategy 1 cannot paraphrase safely)
# =============================================================================

CAT_C_IDS = {
    'PURE_2009 - peazip_S0008',         # may = library might be missing
    'PURE_2007 - nlm_S0185',            # may = access restrictions uncertain
    'PURE_2005 - phin_S0009',           # may = definitional possibility
    'PURE_2004 - e-procurement_S0040',  # may = item may or may not be supplied
    'PURE_2005 - znix_S0000',           # may = application classification hedge
    'PURE_2007 - ertms_S0003',          # may = TSI may define conditions
    'PURE_2010 - gparted_S0012',        # may = feature in a future release
    'PURE_2004 - ijis_S0006',           # may = procedural repetition uncertain
    'PURE_2007 - eirene fun 7_S0127',   # may = possible sources enumerated
    'PURE_2005 - clarus high_S0033',    # may = archiving decision deferred
    'PURE_2001 - elsfork_S0065',        # may = contingent shutdown possibility
    'PROMISE_18_0063',                  # may = system may be able to include
}

# =============================================================================
# HAND-CRAFTED INVARIANCE PROBES — Category A and B items (40 total)
#
# Format: item_id -> (probe_text, operation_label)
#
# Operation labels:
#   rule_canonical_permission : can/may -> "is permitted to"
#   rule_canonical_ability    : can     -> "is able to"
#   rule_nominalization       : verb phrase -> gerund/noun phrase as subject
#   rule_active_restructure   : reorder clauses, change voice without synonym
#   rule_adjectival_passive   : passive verb phrase -> adjectival form
#
# Semantic invariant maintained: modal meaning (permission/capability/possibility)
# and propositional content are preserved across all paraphrases.
# =============================================================================

HAND_CRAFTED_PROBES = {

    # ── Category A: Permission (actor is permitted to do something) ──────────

    'PURE_0000 - inventory_S0005': (
        'Inventory administrators are permitted to be assigned to any administrative '
        'or organizational level according to their tasks.',
        'rule_canonical_permission'
    ),
    'PURE_2009 - peppol_S0562': (
        'Responsibility for composing and supplying such a dossier is permitted to be '
        'distributed across different organizations in different countries, '
        'provided an agreed structure is in place.',
        'rule_canonical_permission'
    ),
    'PURE_2009 - peazip_S0023': (
        'Using PeaZip, users are able to decompress the contents of a selected '
        'compressed archive and extract them into a folder.',
        'rule_canonical_ability'
    ),
    'PURE_0000 - inventory_S0004': (
        'Inventory administrators are users who are permitted to be delegated by any '
        'administrative level; their assigned permissions vary depending on their tasks.',
        'rule_canonical_permission'
    ),
    'PURE_2008 - peering_S0011': (
        "Users are also permitted to receive preferential treatment as determined "
        "by the policy associated with a particular provider's business logic.",
        'rule_canonical_permission'
    ),
    'PURE_2007 - eirene fun 7_S0107': (
        'The railways are permitted to employ various types of call restriction '
        'as an additional security measure.',
        'rule_canonical_permission'
    ),
    'PURE_2007 - puget sound_S0024': (
        'Course administrators are not required to provide web feeds '
        'for every page in their course.',
        'rule_active_restructure'
    ),
    'PURE_2006 - eirene sys 15_S0165': (
        'Location-dependent addressing is permitted to be provided through '
        'cell-dependent routing or by using location information from external sources.',
        'rule_canonical_permission'
    ),
    'PURE_2002 - evla back_S0033': (
        'A small number of authorized individuals are permitted to access '
        'parts of the system that are ordinarily restricted.',
        'rule_canonical_permission'
    ),
    'PURE_2008 - virtual ed_S0017': (
        'Users are able to collaborate on a single document in real time and share '
        'their work with other users through secure file sharing methods and protocols.',
        'rule_canonical_ability'
    ),
    'PURE_0000 - inventory_S0010': (
        'Inter-faculty transfer requests are permitted to be submitted by any '
        'authorised user and require approval from the faculty group or a higher authority.',
        'rule_canonical_permission'
    ),
    'PURE_0000 - inventory_S0046': (
        'The system supports creation of three report types: a User Permission Report, '
        'a Request Report, and an Assets By Location Report.',
        'rule_active_restructure'
    ),
    'PURE_0000 - inventory_S0017': (
        'Requests to borrow an asset or reserve a location are permitted to be '
        'submitted by any authorised user.',
        'rule_canonical_permission'
    ),
    'PURE_2004 - e-procurement_S0005': (
        'Purchase Order variants are permitted to support revision or cancellation '
        'of content and fulfilment terms including delivery details and required dates.',
        'rule_canonical_permission'
    ),
    'PROMISE_8_0037': (
        'Streaming a movie is permitted only when the customer has purchased it '
        'and the 2-day streaming window has not elapsed.',
        'rule_canonical_permission'
    ),

    # ── Category B: Capability (system or component is able to do something) ─

    'PURE_2010 - split merge_S0001': (
        'Viewing PDF files is supported on almost any platform including Macintosh, '
        'Windows, UNIX, LINUX, and numerous mobile platforms, '
        'though file manipulation is typically not free.',
        'rule_nominalization'
    ),
    'PURE_2001 - libra_S0045': (
        'The scheduler evaluates whether completing the job within the requisite '
        'deadline is feasible, taking into account execution time and the status '
        'of other pending jobs on various nodes.',
        'rule_active_restructure'
    ),
    'PURE_1999 - multi-mahjong_S0041': (
        'The validity of these requirements is to be determined during the design '
        'phase, and as a result most of them are classified as Level 2 or Level 3.',
        'rule_active_restructure'
    ),
    'PURE_0000 - cctns_S0005': (
        'When properly configured, the system supports creating, updating, and '
        'deleting information such as acts, sections, state-specific data, '
        'castes, tribes, and property information.',
        'rule_active_restructure'
    ),
    'PURE_2001 - ctc network_S0062': (
        'The Web Map application generates a map that is displayable '
        'on an Internet WWW server.',
        'rule_adjectival_passive'
    ),
    'PURE_0000 - cctns_S0090': (
        'Longer pages are more appropriate in cases where users prefer uninterrupted '
        'reading or where the page is required to correspond to a paper counterpart.',
        'rule_active_restructure'
    ),
    'PURE_2007 - nlm_S0231': (
        'Content delivery is supported through XSLT, CSS style sheets, or other '
        'interface mechanisms for web display or other means.',
        'rule_nominalization'
    ),
    'PURE_2009 - inventory 2.0_S0127': (
        'When the user clicks add item, the inventory matrix is presented so that '
        'an item category is selectable before proceeding to the item details screen.',
        'rule_active_restructure'
    ),
    'PURE_2001 - elsfork_S0137': (
        'Defining a data exchange format between wind power plants and a control '
        'system, such as the Open Windmill Exchange Format, is supported.',
        'rule_nominalization'
    ),
    'PURE_2008 - viper_S0043': (
        'Although the system is not required to interact with other interfaces, '
        'customization of the system is supported.',
        'rule_active_restructure'
    ),
    'PURE_2008 - keepass_S0039': (
        'Passwords stored in the database are transferable to website accounts '
        'and applications securely, without the need to retype them.',
        'rule_adjectival_passive'
    ),
    'PURE_2001 - ctc network_S0007': (
        'The software design enables multiple instances of a building block to be '
        'deployed within a specific agency simply through configuration.',
        'rule_active_restructure'
    ),
    'PURE_2007 - water use_S0026': (
        'In the GIS, the UID has a physical location association and is therefore '
        'representable within a data layer for WUPs, stream gauges, or rainfall stations.',
        'rule_adjectival_passive'
    ),
    'PURE_2009 - warc III_S0084': (
        'The basic report form resembles what the crawlers produce, making it useful '
        'as a substitute when a crawler report is unavailable for a collection.',
        'rule_active_restructure'
    ),
    'PURE_2009 - peppol_S0087': (
        'The contracting authority is able to perform the same process to verify '
        'whether submitted attestations satisfy the required criteria.',
        'rule_canonical_ability'
    ),
    'PURE_2009 - peazip_S0018': (
        'Through PeaZip, users are able to access multiple computer system management '
        'tools covering both storage units and the system itself.',
        'rule_canonical_ability'
    ),
    'PURE_2009 - video search_S0046': (
        'When the user activates the streaming host option, the system initiates a '
        'query of video hosting sites in its database at the start of each search.',
        'rule_active_restructure'
    ),
    'PURE_2008 - peering_S0009': (
        'End-users are assignable via DNS through regular updates by peering agents '
        'of participating CDNs, or alternatively through redirection at the CDN gateway.',
        'rule_adjectival_passive'
    ),
    'PURE_2009 - peazip_S0006': (
        'Applying more than one feature to a single archive or file simultaneously is supported.',
        'rule_nominalization'
    ),
    'PURE_2004 - e-procurement_S0034': (
        'A Digital Signature is an electronic signature that is applicable to the '
        'entire Document payload, excluding the signature element, '
        'using the W3C Digital Signature standard.',
        'rule_adjectival_passive'
    ),
    'PURE_2004 - e-procurement_S0014': (
        'Internal instance identifiers are usable to prevent repetition of the same '
        'set of information across multiple locations within a Document.',
        'rule_adjectival_passive'
    ),
    'PURE_2008 - keepass_S0109': (
        'The Composite Master Key feature combines a master password with key files; '
        'all components are required to be present before the database is unlockable.',
        'rule_active_restructure'
    ),
    'PURE_2005 - triangle_S0005': (
        'Adding new games to the software is supported without requiring '
        'modifications to the original application code.',
        'rule_nominalization'
    ),
    'PURE_2008 - keepass_S0003': (
        'KeePass Password Safe is a compact system that is portable between '
        'computers via a USB stick.',
        'rule_adjectival_passive'
    ),
    'PURE_2002 - evla back_S0032': (
        'Remote access to the system is available to the software developer, '
        'enabling troubleshooting outside of EVLA working hours.',
        'rule_nominalization'
    ),
}

# =============================================================================
# NFR MINI-BENCHMARK — 18 probes, 8 NFR categories
# Stored as a standalone file, NOT injected into SRS-ProbeCore-v5
# =============================================================================

NFR_MINIBENCH = {
    'name': 'NFR-ShortcutMiniBench-v1',
    'description': (
        'Standalone supplementary benchmark of 18 NFR-flavoured shortcut probes. '
        'Each probe inserts a non-functional requirement distractor into a clean '
        'functional requirement. A reliable corrector must not absorb the NFR clause '
        'into its structural rewrite. Reported separately from SRS-ProbeCore-v5 '
        'to avoid scoring ambiguity in the main benchmark. '
        'See Choice 3 rationale in project notes.'
    ),
    'total_probes': 18,
    'nfr_categories': [
        'performance_trigger',
        'security_trigger',
        'availability_trigger',
        'fault_tolerance_trigger',
        'scalability_trigger',
        'usability_trigger',
        'maintainability_trigger',
        'compliance_trigger',
    ],
    'scoring_criterion': (
        'model does NOT introduce the NFR category implied by the distractor '
        'into the structural rewrite of the base requirement'
    ),
    'probes': [
        # ── Performance (3) ──────────────────────────────────────────────────
        {
            'probe_id': 'NFR_SHC_1',
            'nfr_category': 'performance_trigger',
            'source_item_id': 'PROMISE_17_0056',
            'base_text': 'The system shall validate the amount is a multiple of $20.',
            'probe_text': (
                'The system shall validate the amount is a multiple of $20 '
                'within 200 milliseconds.'
            ),
            'distractor': 'within 200 milliseconds',
            'hallucination_risk': (
                'hallucinate a response-time constraint into the validation rewrite'
            ),
            'scoring': {
                'criterion': 'model does NOT introduce any performance or timing clause',
                'pass': 1, 'fail': 0
            },
        },
        {
            'probe_id': 'NFR_SHC_2',
            'nfr_category': 'performance_trigger',
            'source_item_id': 'PROMISE_45_0101',
            'base_text': (
                'The system shall authenticate users at the beginning of each session.'
            ),
            'probe_text': (
                'The system shall authenticate users at the beginning of each session '
                'with a maximum latency of 500ms.'
            ),
            'distractor': 'with a maximum latency of 500ms',
            'hallucination_risk': (
                'hallucinate a latency bound into the authentication rewrite'
            ),
            'scoring': {
                'criterion': 'model does NOT introduce any latency or timing clause',
                'pass': 1, 'fail': 0
            },
        },
        {
            'probe_id': 'NFR_SHC_3',
            'nfr_category': 'performance_trigger',
            'source_item_id': 'PROMISE_4_0018',
            'base_text': (
                'The Disputes System must be accessible by both internal and external users.'
            ),
            'probe_text': (
                'The Disputes System must be accessible by both internal and external users, '
                'supporting up to 500 concurrent users.'
            ),
            'distractor': 'supporting up to 500 concurrent users',
            'hallucination_risk': (
                'hallucinate a concurrent-user load constraint into the access rewrite'
            ),
            'scoring': {
                'criterion': 'model does NOT introduce any concurrency or load clause',
                'pass': 1, 'fail': 0
            },
        },
        # ── Security (3) ─────────────────────────────────────────────────────
        {
            'probe_id': 'NFR_SHC_4',
            'nfr_category': 'security_trigger',
            'source_item_id': 'PROMISE_20_0065',
            'base_text': (
                'Users shall be required to log in to the Cafeteria Ordering System '
                'for all operations except viewing a menu.'
            ),
            'probe_text': (
                'Users shall be required to log in to the Cafeteria Ordering System '
                'for all operations except viewing a menu, using two-factor authentication.'
            ),
            'distractor': 'using two-factor authentication',
            'hallucination_risk': (
                'hallucinate a specific authentication mechanism into the login rewrite'
            ),
            'scoring': {
                'criterion': (
                    'model does NOT introduce any authentication mechanism '
                    'or security protocol'
                ),
                'pass': 1, 'fail': 0
            },
        },
        {
            'probe_id': 'NFR_SHC_5',
            'nfr_category': 'security_trigger',
            'source_item_id': 'PROMISE_17_0058',
            'base_text': (
                'The system shall validate the amount is available in the user account '
                'before releasing funds to the user.'
            ),
            'probe_text': (
                'The system shall validate the amount is available in the user account '
                'before releasing funds to the user over an encrypted channel.'
            ),
            'distractor': 'over an encrypted channel',
            'hallucination_risk': (
                'hallucinate a transport encryption constraint into the financial '
                'validation rewrite'
            ),
            'scoring': {
                'criterion': (
                    'model does NOT introduce any encryption or transport security clause'
                ),
                'pass': 1, 'fail': 0
            },
        },
        {
            'probe_id': 'NFR_SHC_6',
            'nfr_category': 'security_trigger',
            'source_item_id': 'PROMISE_32_0079',
            'base_text': (
                'The user must be able to set an option to hide information based on '
                'who has tagged it.'
            ),
            'probe_text': (
                'The user must be able to set an option to hide information based on '
                'who has tagged it, in compliance with the access control policy.'
            ),
            'distractor': 'in compliance with the access control policy',
            'hallucination_risk': (
                'hallucinate a policy compliance clause into the user-preference rewrite'
            ),
            'scoring': {
                'criterion': (
                    'model does NOT introduce any policy compliance or '
                    'access control reference'
                ),
                'pass': 1, 'fail': 0
            },
        },
        # ── Availability (2) ─────────────────────────────────────────────────
        {
            'probe_id': 'NFR_SHC_7',
            'nfr_category': 'availability_trigger',
            'source_item_id': 'PROMISE_16_0051',
            'base_text': (
                'The system shall support the ability to perform a send only operation.'
            ),
            'probe_text': (
                'The system shall support the ability to perform a send only operation '
                'with 99.9% uptime guarantee.'
            ),
            'distractor': 'with 99.9% uptime guarantee',
            'hallucination_risk': (
                'hallucinate an availability SLA into the capability rewrite'
            ),
            'scoring': {
                'criterion': (
                    'model does NOT introduce any availability percentage or SLA clause'
                ),
                'pass': 1, 'fail': 0
            },
        },
        {
            'probe_id': 'NFR_SHC_8',
            'nfr_category': 'availability_trigger',
            'source_item_id': 'PURE_2007 - ertms_S0018',
            'base_text': (
                'E ETCS trainborne equipment shall be capable of receiving information '
                'from the national train control system.'
            ),
            'probe_text': (
                'E ETCS trainborne equipment shall be capable of receiving information '
                'from the national train control system at all times including degraded '
                'mode operation.'
            ),
            'distractor': 'at all times including degraded mode operation',
            'hallucination_risk': (
                'hallucinate a continuous availability and degraded-mode constraint '
                'into the capability rewrite'
            ),
            'scoring': {
                'criterion': (
                    'model does NOT introduce any continuous-availability or '
                    'degraded-mode clause'
                ),
                'pass': 1, 'fail': 0
            },
        },
        # ── Fault tolerance (2) ───────────────────────────────────────────────
        {
            'probe_id': 'NFR_SHC_9',
            'nfr_category': 'fault_tolerance_trigger',
            'source_item_id': 'PROMISE_2_0008',
            'base_text': (
                'The system shall be able to call the seller or buyer to schedule '
                'an appointment.'
            ),
            'probe_text': (
                'The system shall be able to call the seller or buyer to schedule '
                'an appointment, with automatic retry on connection failure.'
            ),
            'distractor': 'with automatic retry on connection failure',
            'hallucination_risk': (
                'hallucinate a fault-tolerance retry mechanism into the calling '
                'requirement rewrite'
            ),
            'scoring': {
                'criterion': (
                    'model does NOT introduce any retry, recovery, or '
                    'fault-tolerance clause'
                ),
                'pass': 1, 'fail': 0
            },
        },
        {
            'probe_id': 'NFR_SHC_10',
            'nfr_category': 'fault_tolerance_trigger',
            'source_item_id': 'PROMISE_9_0042',
            'base_text': (
                'The leads washing functionality will validate all leads received by '
                'the web service for valid data.'
            ),
            'probe_text': (
                'The leads washing functionality will validate all leads received by '
                'the web service for valid data and log all failures to a persistent '
                'error store.'
            ),
            'distractor': 'and log all failures to a persistent error store',
            'hallucination_risk': (
                'hallucinate an error-logging and persistence clause into the '
                'validation rewrite'
            ),
            'scoring': {
                'criterion': (
                    'model does NOT introduce any error logging or persistence clause'
                ),
                'pass': 1, 'fail': 0
            },
        },
        # ── Scalability (2) ──────────────────────────────────────────────────
        {
            'probe_id': 'NFR_SHC_11',
            'nfr_category': 'scalability_trigger',
            'source_item_id': 'PROMISE_47_0107',
            'base_text': (
                'The system shall allow the user to register for an account from '
                'any page in the system.'
            ),
            'probe_text': (
                'The system shall allow the user to register for an account from '
                'any page in the system without degradation under peak load.'
            ),
            'distractor': 'without degradation under peak load',
            'hallucination_risk': (
                'hallucinate a scalability constraint into the registration capability '
                'rewrite'
            ),
            'scoring': {
                'criterion': (
                    'model does NOT introduce any load, scaling, or peak-traffic clause'
                ),
                'pass': 1, 'fail': 0
            },
        },
        {
            'probe_id': 'NFR_SHC_12',
            'nfr_category': 'scalability_trigger',
            'source_item_id': 'PROMISE_8_0039',
            'base_text': (
                'Website shall allow customers to add their own movie review for a '
                'selected movie.'
            ),
            'probe_text': (
                'Website shall allow customers to add their own movie review for a '
                'selected movie for up to 10,000 simultaneous submissions.'
            ),
            'distractor': 'for up to 10,000 simultaneous submissions',
            'hallucination_risk': (
                'hallucinate a concurrent submission capacity constraint into the '
                'contribution rewrite'
            ),
            'scoring': {
                'criterion': (
                    'model does NOT introduce any concurrency limit or scale bound'
                ),
                'pass': 1, 'fail': 0
            },
        },
        # ── Usability (2) ────────────────────────────────────────────────────
        {
            'probe_id': 'NFR_SHC_13',
            'nfr_category': 'usability_trigger',
            'source_item_id': 'PROMISE_33_0083',
            'base_text': 'The system should ask the user to key in the pickup date.',
            'probe_text': (
                'The system should ask the user to key in the pickup date within no '
                'more than two interaction steps.'
            ),
            'distractor': 'within no more than two interaction steps',
            'hallucination_risk': (
                'hallucinate an interaction-step efficiency constraint into the '
                'data-entry rewrite'
            ),
            'scoring': {
                'criterion': (
                    'model does NOT introduce any interaction-step count or '
                    'usability efficiency clause'
                ),
                'pass': 1, 'fail': 0
            },
        },
        {
            'probe_id': 'NFR_SHC_14',
            'nfr_category': 'usability_trigger',
            'source_item_id': 'PROMISE_45_0098',
            'base_text': (
                'The system shall allow the user to modify the date preference set '
                'of an already submitted meeting response.'
            ),
            'probe_text': (
                'The system shall allow the user to modify the date preference set '
                'of an already submitted meeting response through an intuitive and '
                'self-explanatory interface.'
            ),
            'distractor': 'through an intuitive and self-explanatory interface',
            'hallucination_risk': (
                'hallucinate a UI quality attribute into the modification capability '
                'rewrite'
            ),
            'scoring': {
                'criterion': (
                    'model does NOT introduce any UI quality or usability attribute'
                ),
                'pass': 1, 'fail': 0
            },
        },
        # ── Maintainability (2) ───────────────────────────────────────────────
        {
            'probe_id': 'NFR_SHC_15',
            'nfr_category': 'maintainability_trigger',
            'source_item_id': 'PROMISE_7_0034',
            'base_text': (
                'The System shall generate Inventory Quantity Adjustment document '
                'automatically when daily Product Sales data is available.'
            ),
            'probe_text': (
                'The System shall generate Inventory Quantity Adjustment document '
                'automatically when daily Product Sales data is available, without '
                'requiring changes to the core application code.'
            ),
            'distractor': 'without requiring changes to the core application code',
            'hallucination_risk': (
                'hallucinate a maintainability constraint into the document-generation '
                'rewrite'
            ),
            'scoring': {
                'criterion': (
                    'model does NOT introduce any code-change restriction or '
                    'maintainability clause'
                ),
                'pass': 1, 'fail': 0
            },
        },
        {
            'probe_id': 'NFR_SHC_16',
            'nfr_category': 'maintainability_trigger',
            'source_item_id': 'PROMISE_45_0099',
            'base_text': (
                'The system shall allow the initiator to send and receive messages '
                'from users.'
            ),
            'probe_text': (
                'The system shall allow the initiator to send and receive messages '
                'from users using a pluggable messaging module that can be replaced '
                'independently.'
            ),
            'distractor': (
                'using a pluggable messaging module that can be replaced independently'
            ),
            'hallucination_risk': (
                'hallucinate an architectural modularity constraint into the messaging '
                'capability rewrite'
            ),
            'scoring': {
                'criterion': (
                    'model does NOT introduce any modularity, pluggability, or '
                    'architectural constraint'
                ),
                'pass': 1, 'fail': 0
            },
        },
        # ── Compliance / Legal (2) ────────────────────────────────────────────
        {
            'probe_id': 'NFR_SHC_17',
            'nfr_category': 'compliance_trigger',
            'source_item_id': 'PROMISE_20_0066',
            'base_text': (
                'The system shall permit only cafeteria staff members who are on the '
                'list of authorized Menu Managers to create or edit menus.'
            ),
            'probe_text': (
                'The system shall permit only cafeteria staff members who are on the '
                'list of authorized Menu Managers to create or edit menus, in accordance '
                'with data protection regulations.'
            ),
            'distractor': 'in accordance with data protection regulations',
            'hallucination_risk': (
                'hallucinate a regulatory compliance clause into the access-control rewrite'
            ),
            'scoring': {
                'criterion': (
                    'model does NOT introduce any regulatory or legal compliance reference'
                ),
                'pass': 1, 'fail': 0
            },
        },
        {
            'probe_id': 'NFR_SHC_18',
            'nfr_category': 'compliance_trigger',
            'source_item_id': 'PROMISE_5_0024',
            'base_text': (
                'The ratings shall include categories for attempted use of recycled '
                'parts and actual use of recycled parts.'
            ),
            'probe_text': (
                'The ratings shall include categories for attempted use of recycled '
                'parts and actual use of recycled parts, as required by environmental '
                'compliance standards.'
            ),
            'distractor': 'as required by environmental compliance standards',
            'hallucination_risk': (
                'hallucinate an environmental regulatory constraint into the rating '
                'requirement rewrite'
            ),
            'scoring': {
                'criterion': (
                    'model does NOT introduce any environmental or compliance standard '
                    'reference'
                ),
                'pass': 1, 'fail': 0
            },
        },
    ],
}


# =============================================================================
# PIPELINE
# =============================================================================

def run():

    print('=' * 60)
    print('SRS-ProbeCore v5 Build Pipeline')
    print('=' * 60)
    print(f'DRY_RUN = {DRY_RUN}')
    print()

    # Load input
    with open(INPUT_PATH) as f:
        data = json.load(f)

    # Deep copy so we never mutate the loaded object during dry run checks
    working = copy.deepcopy(data)

    print(f'Input loaded : {len(working)} items')
    print(f'Input probes : {sum(i["probe_count"] for i in working)}')
    print()

    # ── PRE-RUN CHECK ────────────────────────────────────────────────────────

    print('── PRE-RUN CHECKS ──────────────────────────────────────────')

    all_ids = {i['item_id'] for i in working}
    two_probe_items = [i for i in working if i['probe_count'] == 2]
    cat_c_in_file   = [i for i in two_probe_items if i['item_id'] in CAT_C_IDS]
    cat_ab_in_file  = [i for i in two_probe_items if i['item_id'] not in CAT_C_IDS]
    hand_crafted_missing = [
        iid for iid in HAND_CRAFTED_PROBES if iid not in all_ids
    ]
    cat_ab_without_map = [
        i for i in cat_ab_in_file if i['item_id'] not in HAND_CRAFTED_PROBES
    ]

    print(f'  2-probe items total     : {len(two_probe_items)}')
    print(f'  Category C (leave alone): {len(cat_c_in_file)}')
    print(f'  Category A+B (inject)   : {len(cat_ab_in_file)}')
    print(f'  HAND_CRAFTED_PROBES map : {len(HAND_CRAFTED_PROBES)} entries')

    if hand_crafted_missing:
        print(f'  WARNING — map entries not in file (will be skipped):')
        for iid in hand_crafted_missing:
            print(f'    {iid}')
    else:
        print(f'  All map entries present in file : OK')

    if cat_ab_without_map:
        print(f'  WARNING — A+B items with no map entry (will NOT get INV_A):')
        for i in cat_ab_without_map:
            print(f'    {i["item_id"]} | modal={i["modal"]}')
    else:
        print(f'  All A+B items covered in map    : OK')

    all_probes_pre = [n for i in working for n in i['probe_neighborhoods']]
    none_gen = [n for n in all_probes_pre if n.get('generation_method') is None]
    print(f'  Probes with gen_method=None : {len(none_gen)}')
    print()

    # ── STEP 2: Invariance probes for Category A+B ───────────────────────────

    print('── STEP 2: Inject invariance probes (Category A+B) ─────────')

    step2_injected = 0
    step2_skipped  = 0

    for item in working:
        if item['probe_count'] != 2:
            continue
        iid = item['item_id']
        if iid in CAT_C_IDS:
            step2_skipped += 1
            continue
        if iid not in HAND_CRAFTED_PROBES:
            continue

        probe_text, operation = HAND_CRAFTED_PROBES[iid]
        new_probe = {
            'probe_id'          : 'INV_A',
            'probe_family'      : 'invariance',
            'probe_text'        : probe_text,
            'expected_relation' : 'stable',
            'operation'         : operation,
            'generation_method' : 'strategy1_structural_paraphrase',
            'scoring': {
                'criterion': (
                    'revised outputs remain semantically aligned and preserve '
                    'the same modal permission or capability meaning'
                ),
                'pass': 1,
                'fail': 0,
            },
            'validation': 'hand_crafted_strategy1',
        }
        # Insert at position 0 — invariance probes stay grouped at the front
        item['probe_neighborhoods'].insert(0, new_probe)
        item['probe_count'] = len(item['probe_neighborhoods'])
        step2_injected += 1

    print(f'  Injected  : {step2_injected}')
    print(f'  Skipped C : {step2_skipped}')
    print()

    # ── STEP 3: Populate generation_method ──────────────────────────────────

    print('── STEP 3: Populate generation_method ──────────────────────')

    step3_fixed = {'directional': 0, 'shortcut': 0}

    for item in working:
        for n in item['probe_neighborhoods']:
            if n.get('generation_method') is not None:
                continue
            if n['probe_family'] == 'directional':
                n['generation_method'] = 'rule_based_modal_substitution'
                step3_fixed['directional'] += 1
            elif n['probe_family'] == 'shortcut':
                n['generation_method'] = 'rule_based_distractor_insertion'
                step3_fixed['shortcut'] += 1

    print(f'  directional probes fixed : {step3_fixed["directional"]}')
    print(f'  shortcut probes fixed    : {step3_fixed["shortcut"]}')
    print()

    # ── STEP 4: Save NFR mini-benchmark ─────────────────────────────────────

    print('── STEP 4: NFR mini-benchmark ──────────────────────────────')
    print(f'  Probes         : {NFR_MINIBENCH["total_probes"]}')
    print(f'  NFR categories : {len(NFR_MINIBENCH["nfr_categories"])}')
    print(f'  Output         : {OUTPUT_NFR}')

    if not DRY_RUN:
        with open(OUTPUT_NFR, 'w') as f:
            json.dump(NFR_MINIBENCH, f, indent=2)
        print(f'  Saved.')
    else:
        print(f'  DRY_RUN — file not written.')
    print()

    # ── STEP 5: Sanity checks and freeze ────────────────────────────────────

    print('── STEP 5: Sanity checks ────────────────────────────────────')

    all_probes = [n for item in working for n in item['probe_neighborhoods']]

    # Check 1: no generation_method=None remaining
    remaining_none = [n for n in all_probes if n.get('generation_method') is None]
    status = 'PASS' if not remaining_none else f'FAIL ({len(remaining_none)} probes)'
    print(f'  generation_method complete : {status}')

    # Check 2: probe_count matches actual neighborhood length
    count_mismatch = [
        i for i in working
        if i['probe_count'] != len(i['probe_neighborhoods'])
    ]
    status = 'PASS' if not count_mismatch else f'FAIL ({len(count_mismatch)} items)'
    print(f'  probe_count integrity      : {status}')

    # Check 3: all invariance probes have expected_relation=stable
    inv_bad = [
        n for n in all_probes
        if n['probe_family'] == 'invariance'
        and n.get('expected_relation') != 'stable'
    ]
    status = 'PASS' if not inv_bad else f'FAIL ({len(inv_bad)} probes)'
    print(f'  invariance expected_relation: {status}')

    # Check 4: all directional probes have expected_relation=directional
    dir_bad = [
        n for n in all_probes
        if n['probe_family'] == 'directional'
        and n.get('expected_relation') != 'directional'
    ]
    status = 'PASS' if not dir_bad else f'FAIL ({len(dir_bad)} probes)'
    print(f'  directional expected_relation: {status}')

    # Check 5: all shortcut probes have expected_relation=no_shortcut
    shc_bad = [
        n for n in all_probes
        if n['probe_family'] == 'shortcut'
        and n.get('expected_relation') != 'no_shortcut'
    ]
    status = 'PASS' if not shc_bad else f'FAIL ({len(shc_bad)} probes)'
    print(f'  shortcut expected_relation : {status}')

    # Check 6: Category C items still at probe_count=2
    cat_c_counts = [
        i['probe_count'] for i in working if i['item_id'] in CAT_C_IDS
    ]
    all_still_two = all(c == 2 for c in cat_c_counts)
    status = 'PASS' if all_still_two else 'FAIL (some Cat C items were modified)'
    print(f'  Category C untouched       : {status}')

    failures = (
        remaining_none or count_mismatch or inv_bad
        or dir_bad or shc_bad or not all_still_two
    )
    print()

    if failures:
        print('One or more sanity checks FAILED. File not saved. Fix issues above.')
        return

    # Final stats
    families    = Counter([n['probe_family'] for n in all_probes])
    probe_dist  = Counter([i['probe_count'] for i in working])
    sources     = Counter([i['source'] for i in working])
    gen_methods = Counter([n.get('generation_method') for n in all_probes])
    modals      = Counter([i['modal'] for i in working])
    ears        = Counter([i['ears_type'] for i in working])
    inv_ops     = Counter([
        n.get('operation') for n in all_probes
        if n['probe_family'] == 'invariance'
    ])
    cat_c_remaining = [i for i in working if i['probe_count'] == 2]

    print('── FINAL STATS ─────────────────────────────────────────────')
    print(f'  Items             : {len(working)}')
    print(f'  Total probes      : {len(all_probes)}')
    print(f'  Sources           : {dict(sources)}')
    print(f'  Probe families    : {dict(families)}')
    print(f'  Probes per item   : {dict(sorted(probe_dist.items()))}')
    print(f'  Modals            : {dict(modals)}')
    print(f'  EARS types        : {dict(ears)}')
    print(f'  Invariance ops    : {dict(inv_ops)}')
    print(f'  Generation methods: {dict(gen_methods)}')
    print(f'  Cat C remaining   : {len(cat_c_remaining)} items at probe_count=2')
    print()

    if not DRY_RUN:
        with open(OUTPUT_MAIN, 'w') as f:
            json.dump(working, f, indent=2)
        print(f'Frozen: {OUTPUT_MAIN}')
    else:
        print(f'DRY_RUN — no files written. Set DRY_RUN=False to save.')


if __name__ == '__main__':
    run()
