
Code associated with our EMNLP paper "A Generative Joint, Additive, Sequential Model of Topics and Speech Acts in Patient-Doctor Communication". Unfortunately, we have not yet been able to secure IRB approval to release the actual data :(. But we give the expected data format and sample usage below, anyway. 

-Byron Wallace
byron_wallace@brown.edu
http://www.cebm.brown.edu/byron


Data format
--
The basic data format is as follows:

boundary boundary:case_id=BMC3013_1
19 32 42 56 226 309 558 1889 1,2
558 19 58 145 168 216 1,2
56 35 16 20 3,2
…
boundary boundary:case_id=XXX

Where the "boundary" strings demarcate a new session. The last two (comma-separated) entries are the topic and speech act, respectively. 

Sample usage
--
import transcripts
import joint_sequential_SATs

tnb = transcripts.tnb_from_file("data/unigram-cases-joint/train.CRF.speakers.pronoun.question.unigram.joint.0.dat", hold_out_a_set=True)

# train model and make predictions
m = joint_sequential_SATs.JointSequential(tnb)
m.estimate_parameters() # may take a while...
test_cases = transcripts.load_test_cases("data/unigram-cases-joint/test.CRF.speakers.pronoun.question.unigram.joint.0.dat", tnb)
preds_Y, preds_S = m.predict_set_sequential_joint(test_cases)

# now assess performance 
import process_results
test_Y, test_S = transcripts.parse_labels_file("data/unigram-cases-joint/test.CRF.speakers.pronoun.question.unigram.joint.0_labels.dat")
print process_results.calc_metrics(test_Y, test_S, preds_Y, preds_S)