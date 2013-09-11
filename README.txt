import transcripts
import joint_sequential_SATs
tnb = transcripts.tnb_from_file("data/unigram-cases-joint/train.CRF.speakers.pronoun.question.unigram.joint.0.dat", hold_out_a_set=True)

# train model and make predictions
m = joint_sequential_SATs.JointSequential(tnb)
m.estimate_parameters()
test_cases = transcripts.load_test_cases("data/unigram-cases-joint/test.CRF.speakers.pronoun.question.unigram.joint.0.dat", tnb)
preds_Y, preds_S = m.predict_set_sequential_joint(test_cases)

# now assess performance
import process_results
test_Y, test_S = transcripts.parse_labels_file("data/unigram-cases-joint/test.CRF.speakers.pronoun.question.unigram.joint.0_labels.dat")
print process_results.calc_metrics(test_Y, test_S, preds_Y, preds_S)