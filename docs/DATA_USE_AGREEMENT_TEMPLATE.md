# Mandarin-sEEG Restricted Data Use Agreement Template

> **Draft for institutional review.** Replace every bracketed field and obtain
> approval from the responsible ethics, legal, information-security, and data
> governance offices before using this document to grant access. This template
> is not itself an open-data licence and is not activated by downloading the
> public code repository.

**Dataset:** *Mandarin-sEEG: A Large-Scale Dataset for Speech Articulation*  
**Version of this agreement:** 3.0  
**Dataset version covered:** [v1.0 or applicable version]  
**Data provider:** Westlake University and Zhejiang University School of Medicine Affiliated Hospitals  
**Access-request contact:** [email address]  
**Repository or access-request page:** [URL]  
**Date:** [DD Month YYYY]

The following Data Use Terms must be signed, submitted, and approved before an investigator may receive access to Restricted Data from the Mandarin-sEEG dataset. The Dataset contains human stereotactic electroencephalography (sEEG) recordings and speech-related recordings. Speech audio and combinations of neurophysiological, electrode, linguistic, and clinical information may contain or permit inference of potentially identifying information. These additional precautions are therefore required to protect participants and to maintain the ethical and legal governance of the Dataset.

This Agreement applies to the individual applicant and to the applicant's supervising Principal Investigator (PI), Institution, and approved research team. Access is not granted by signature alone. The application must be approved by the Dataset Access Committee or designated data custodian before data are released.

## A. Definitions of Restricted Data Elements

The following data elements are covered by this Agreement when supplied under a controlled-access application:

| No. | Restricted data element              | Description                                                                                                                                                                                                           |
| --- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | sEEG data                            | Raw or minimally processed intracranial recordings, channel-level data, trial-level signal segments, and associated signal derivatives.                                                                               |
| 2   | Speech and audio data                | Raw or minimally processed speech recordings, audio-derived features, and other materials that may preserve voice identity or speech content.                                                                         |
| 3   | Annotations and task records         | Trial-level timing, phonetic, tonal, word, sentence, behavioral, and task annotations.                                                                                                                                |
| 4   | Electrode and anatomical information | Electrode labels, localization coordinates, implantation information, channel metadata, and related anatomical or clinical information.                                                                               |
| 5   | Participant and session metadata     | Coded demographic, clinical, recording-session, and acquisition information that is not explicitly designated as openly available.                                                                                    |
| 6   | Derived materials                    | Processed signals, features, labels, trained models, model parameters, figures, tables, and other outputs that could reasonably permit reconstruction or inference of a participant, session, or sensitive recording. |

Open metadata, documentation, analysis code, and representative non-identifying examples are governed by their stated repository licenses and are not automatically covered by this table unless they are supplied together with Restricted Data.

## B. Who May Request Access

Investigators are expected to meet one of the following criteria:

1. The applicant is a PI or independent investigator conducting scientific research at a university, hospital, research institute, government agency, or other organization able to provide appropriate research and data-security oversight; or
2. The applicant is a student, postdoctoral researcher, staff scientist, or other trainee working under the supervision of a PI who is named in the application; or
3. The applicant can provide equivalent evidence of a legitimate scientific purpose, appropriate institutional oversight, and adequate technical safeguards, subject to approval by the Dataset Access Committee.

The proposed work must be consistent with participant consent, applicable ethics requirements, and the stated purpose of the access application. The Dataset is intended for non-commercial scientific research, method development, or education. Commercial use, clinical diagnosis, clinical decision-making, and other uses outside the approved application require separate written authorization.

## C. Investigator Obligations

I request access to Restricted Data from the Mandarin-sEEG dataset. I understand that these terms supplement the repository's general data-access conditions and that I must comply with the following obligations.

| No. | Data use term                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Applicant initials |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:------------------:|
| 1   | **Applicant status.** I am a PI or independent investigator as described in Section B.1, or I am applying under the supervision of the PI named below. If I am not a PI, I understand that my PI must review and approve this application and that I may not access the Dataset until my individual access has been approved. **Name of supervising PI:** __________________________________________                                                                                                 | ______             |
| 2   | **Approved purpose.** I will use the Restricted Data only for the scientific, methodological, or educational purpose, data categories, users, and access period described in my approved application. I will not use the Dataset for clinical diagnosis, treatment decisions, employment or insurance decisions, law-enforcement decisions, or other purposes outside the approved scope.                                                                                                            | ______             |
| 3   | **No re-identification.** I will not attempt to identify or re-identify any participant, patient, family member, clinician, or other person represented in the Dataset. This includes attempts based on voice characteristics, speech content, electrode locations, clinical information, timestamps, or combinations of Dataset elements.                                                                                                                                                           | ______             |
| 4   | **No contact.** I will not contact or attempt to contact any participant or other person represented in the Dataset.                                                                                                                                                                                                                                                                                                                                                                                 | ______             |
| 5   | **No data linkage.** I will not link the Dataset with another dataset, registry, database, public source, or private record when the linkage could increase the risk of participant or session identification.                                                                                                                                                                                                                                                                                       | ______             |
| 6   | **No redistribution.** I will not copy, redistribute, sell, sublicense, publish, or provide Restricted Data to any other person or organization. I will not share data with members of my team unless they have independently submitted the required information, signed the applicable terms, and received approval as Authorized Users.                                                                                                                                                            | ______             |
| 7   | **Authorized users.** I will maintain an accurate list of all persons who access or process the Dataset. I will not share accounts, passwords, access tokens, or credentials. I will notify the Provider before adding a new user and will ensure that the new user is approved before access is granted.                                                                                                                                                                                            | ______             |
| 8   | **Data security.** I will store the Restricted Data on an institutionally managed, password-protected system with access limited to Authorized Users. I will use encryption at rest and during transfer whenever technically feasible and required by institutional policy. I will not store the data on an unsecured personal computer, removable drive, public file-sharing service, or unapproved cloud platform.                                                                                 | ______             |
| 9   | **Third-party services.** I will not upload Restricted Data to a public repository, public code platform, general-purpose generative-AI service, or other third-party service without prior written approval from the Provider and appropriate institutional security review.                                                                                                                                                                                                                        | ______             |
| 10  | **Ethics and institutional compliance.** I will comply with all applicable laws, regulations, institutional policies, information-security requirements, and ethics or human-subjects review requirements. I will obtain any required ethics approval, exemption, institutional determination, or other authorization before using the Dataset.                                                                                                                                                      | ______             |
| 11  | **Additional justification.** I will provide additional information or specific justification if requested by the Dataset Access Committee before or during review of my application.                                                                                                                                                                                                                                                                                                                | ______             |
| 12  | **Incident reporting.** I will notify the Provider at [incident-reporting email] without undue delay, and no later than 15 calendar days after becoming aware of suspected or confirmed loss, unauthorized access, disclosure, re-identification, or other violation involving the Dataset. I will cooperate with reasonable containment, investigation, and reporting requirements.                                                                                                                 | ______             |
| 13  | **Deletion or withdrawal.** I will stop using the Dataset and delete all copies under my control within [30] calendar days after project completion, expiry or termination of access, or a written request from the Provider, except for copies that must be retained by law or institutional policy. Any retained copy will remain subject to this Agreement.                                                                                                                                       | ______             |
| 14  | **Release of derived materials.** I may publish aggregate findings, non-identifying figures, and analysis code only when they do not disclose Restricted Data or create a reasonable risk of participant or session identification. I will not publicly release raw audio, identifiable or near-identifiable voice samples, trial-level records, or sensitive individual-level maps. I will consult the Provider before releasing a Derived Material that may contain participant-level information. | ______             |
| 15  | **Citation and acknowledgement.** Any paper, abstract, thesis, presentation, software release, or other public report using the Dataset will cite the Dataset descriptor and acknowledge the participants, clinical teams, data collectors, curators, providers, and applicable funding sources. I will replace the temporary citation below with the final Data Descriptor citation when it becomes available.                                                                                      | ______             |
| 16  | **Authorship.** I understand that access to or use of the Dataset does not by itself create an entitlement to authorship for the Provider, dataset curators, or data collectors. Authorship, if appropriate, will be determined under applicable authorship standards or a separate written collaboration agreement.                                                                                                                                                                                 | ______             |
| 17  | **Consequences of non-compliance.** I understand that failure to comply may result in suspension or termination of access, a request to delete data, notification of my Institution and relevant oversight bodies, and other action permitted by applicable law or policy.                                                                                                                                                                                                                           | ______             |
| 18  | **Data limitations.** I understand that the Dataset is provided for research use on an "AS IS" basis and is not validated or authorized for clinical diagnosis or treatment. To the extent permitted by applicable law, the Provider makes no warranty regarding completeness, accuracy, fitness for a particular purpose, uninterrupted availability, or clinical suitability.                                                                                                                      | ______             |

## D. Access Application Information

The following information is required for every applicant. Each person who will access Restricted Data must submit an individual application, even when working in the same research group.

| Field                                               | Applicant response |
| --------------------------------------------------- | ------------------ |
| Full name                                           |                    |
| Appointment or title                                |                    |
| Highest academic degree                             |                    |
| Institutional affiliation                           |                    |
| Institutional email                                 |                    |
| Telephone                                           |                    |
| Supervising PI, if applicable                       |                    |
| PI email                                            |                    |
| Research project title                              |                    |
| Scientific purpose and planned analyses             |                    |
| Requested data elements and scope                   |                    |
| Ethics approval, exemption, or determination number |                    |
| Funding source and commercial-use declaration       |                    |
| Proposed storage environment                        |                    |
| Requested access start and end dates                |                    |
| Names of other proposed users in the research group |                    |
| Additional information or justification             |                    |

## E. Applicant Certification

I certify that the information in this application is accurate and complete. I have read and understood this Agreement. I will use the Restricted Data only as approved, will ensure that all relevant team members comply with these terms, and will promptly report any suspected breach or unauthorized use.

**Applicant name:** _________________________________________________  
**Signature:** _____________________________________________________  
**Date:** _________________________________________________________

## F. Supervising PI Certification

I confirm that the applicant is working under my supervision, where applicable, and that the proposed work has legitimate scientific or educational purpose. I will ensure that all users in my group independently obtain approval, receive appropriate training, and comply with this Agreement.

**PI name:** ______________________________________________________  
**Institution:** ___________________________________________________  
**Email:** ________________________________________________________  
**Signature:** ____________________________________________________  
**Date:** _________________________________________________________

## G. Institutional Representative, if required

**Name and title:** _________________________________________________  
**Institution:** ___________________________________________________  
**Signature:** ____________________________________________________  
**Date:** _________________________________________________________

## H. Dataset Access Committee Approval

| Decision                 | Selection or information                                     |
| ------------------------ | ------------------------------------------------------------ |
| Review outcome           | [ ] Approved  [ ] Approved with conditions  [ ] Not approved |
| Approved data categories |                                                              |
| Approved users           |                                                              |
| Access start date        |                                                              |
| Access end date          |                                                              |
| Conditions or comments   |                                                              |
| Reviewer name and title  |                                                              |
| Reviewer signature       |                                                              |
| Date                     |                                                              |

Applications should be submitted as a signed PDF to **[access-request email]** or through **[access-request URL]**. The Provider may request clarification or additional documentation. Access will not be enabled until the application has been approved and all required signatures have been received.

## Suggested Dataset Citation

> Mandarin-sEEG Research Group (2026). *Mandarin-sEEG: A Large-Scale Dataset for Speech Articulation*, version [X.X]. [Repository URL or DOI].

## Suggested Acknowledgement

> We used data from the Mandarin-sEEG dataset for speech and language research. We thank the participants, clinical teams, data collectors, and dataset curators. Data access was provided under a controlled-access Data Use Agreement. This work was supported by the STI2030-Major Project (2022ZD0208805), the National Natural Science Foundation of China (Grant No. 623B2085), and the "Pioneer" and "Leading Goose" R&D Program of Zhejiang (2024C03002), where applicable.

The funding statement should be adjusted to reflect the funding actually associated with the Dataset and the Recipient's project.


