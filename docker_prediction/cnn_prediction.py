import json
import base64
import requests
import pickle
import tensorflow as tf


MAX_SEQUENCE_LENGTH = 500


def argmax(lst):
    result = []
    for item in lst:
        if item[0] >= item[1]:
            result.append(0)
        else:
            result.append(1)
    return result

# Modify the name of your model (`hv_grid` here) to match what you used in Section 2
server_endpoint = 'http://localhost:8501/v1/models/cnn_model:predict'

urgent=[
    'This 5-year-old male presents to Children’s Hospital Emergency Department by the mother with "have asthma." Mother states he has been wheezing and coughing. They saw their primary medical doctor. He was evaluated at the clinic, given the breathing treatment and discharged home, was not having asthma, prescribed prednisone and an antibiotic. They told to go to the ER if he got worse. He has had some vomiting and some abdominal pain. His peak flows on the morning are normal at 150, but in the morning, they were down to 100 and subsequently decreased to 75 over the course of the day.',
    "End-stage renal disease (ESRD).,DISCHARGE DIAGNOSIS: , End-stage renal disease (ESRD).,PROCEDURE:,  Cadaveric renal transplant.,HISTORY OF PRESENT ILLNESS: , This is a 46-year-old gentleman with end-stage renal disease (ESRD) secondary to diabetes and hypertension, who had been on hemodialysis since 1993 and is also status post cadaveric kidney transplant in 1996 with chronic rejection.,PAST MEDICAL HISTORY:  ,1.  Diabetes mellitus diagnosed 12 year ago.,2.  Hypertension.,3.  Coronary artery disease with a myocardial infarct in September of 2006.,4.  End-stage renal disease.,PAST SURGICAL HISTORY: , Coronary artery bypass graft x5 in 1995 and cadaveric renal transplant in 1996.,SOCIAL HISTORY:  ,The patient denies tobacco or ethanol use.,FAMILY HISTORY:,  Hypertension.,PHYSICAL EXAMINATION:  ,GENERAL:  The patient wa alert and oriented x3 in no acute distress, healthy-appearing male.,VITAL SIGNS:  Temperature 96.6, blood pressure 166/106, heart rate 83, respiratory rate 18, and saturation 96% on room air.,CARDIOVASCULAR:  Regular rate and rhythm.,PULMONARY:  Clear to auscultation bilaterally.,ABDOMEN:  Soft, nontender, and nondistended with positive bowel sounds.,EXTREMITIES:  No clubbing, cyanosis, or edema.,PERTINENT LABORATORY DATA: , White blood cell count 6.4, hematocrit 34.6, and platelet count 182.  Sodium 137, potassium 5.4, BUN 41, creatinine 7.9, and glucose 295.  Total protein 6.5, albumin 3.4, AST 51, ALT 51, alk phos 175, and total bilirubin 0.5.,COURSE IN HOSPITAL: , The patient wa admitted postoperatively to the surgical intensive care unit.  Initially, the patient had a decrease in hematocrit from 30 to 25.  The patient's hematocrit stabilized at 25.  During the patient's stay, the patient's creatinine progressively decreased from 8.1 to a creatinine at the time of discharge of 2.3.  The patient wa making excellent urine throughout his stay.  The patient's Jackson-Pratt drain wa removed on postoperative day #1 and he wa moved to the floor.  The patient wa advanced in diet appropriately.  The patient wa started on Prograf by postoperative day #2.  Initial Prograf level came back high at 18.  The patient's Prograf dos were changed accordingly and today, the patient is deemed stable to be discharged home.  During the patient's stay, the patient received four total dos of Thymoglobulin.  Today, he will complete his final dose of Thymoglobulin prior to being discharged.  In addition, today, the patient ha an elevated blood pressure of 198/96.  The patient is being given an extra dose of metoprolol for this blood pressure.  In addition, the patient ha an elevated glucose of 393 and for this reason he ha been given an extra dose of insulin.  These lab will be rechecked later today and once his blood pressure ha decreased to systolic blood pressure le than 116 and his glucose ha come down to a more normal level, he will be discharged to home.,DISCHARGE INSTRUCTIONS: , The patient is discharged with instruction to seek medical attention in the event if he develops fevers, chills, nausea, vomiting, decreased urine output, or other concerns.  He is discharged on a low-potassium diet with activity a tolerated.  He is instructed that he may shower; however, he is to undergo no underwater soaking activity for approximately two weeks.  The patient will be followed up in the Transplant Clinic at ABCD tomorrow, at which time, his lab will be rechecked.  The patients Prograf level at the time of discharge are pending; however, given that his Prograf dose wa decreased, he will be followed tomorrow at the Renal Transplant Clinic."
]
non_urgent=[
    'Mr. XYZ is 41 years of age, who works for Chevron and lives in Angola. He was playing basketball in Angola back last Wednesday, Month DD, YYYY, when he was driving toward the basket and felt a pop in his posterior leg. He was seen locally and diagnosed with an Achilles tendon rupture. He has been on crutches and has been nonweightbearing since that time. He had no pain prior to his injury. He has had some swelling that is mild. He has just been on aspirin a day due to his traveling time. Pain currently is minimal.',
    'This is a 56-year-old female who fell on November 26, 2007 at 11:30 a.m. while at work. She did not recall the specifics of her injury but she thinks that her right foot inverted and subsequently noticed pain in the right ankle. She describes no other injury at this time.'
]

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

request = urgent + non_urgent
requests_tokenized = tokenizer.texts_to_sequences(request)
requests_tokenized = tf.keras.preprocessing.sequence.pad_sequences(requests_tokenized,maxlen=MAX_SEQUENCE_LENGTH)
request_data = requests_tokenized.tolist()

# Create payload request 
payload = json.dumps({"instances": request_data})

response = requests.post(server_endpoint, data=payload)
predictions = json.loads(response.content)
print('native response print: ')
print(predictions)
prediction_labels = argmax(predictions['predictions'])
print(prediction_labels)



