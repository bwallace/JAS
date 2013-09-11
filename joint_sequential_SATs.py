'''
A joint, additive model for speech acts and topic codes
(and transitions). See paper:

"A Generative Joint, Additive, Sequential Model of Topics and 
Speech Acts in Patient-Doctor Communication". 
Byron C. Wallace, Thomas A. Trikalinos, M. Barton Laws, 
Ira B. Wilson and Eugene Charniak. EMNLP 2013. 

for discussion of the model. Parameter inference is done via Newton 
optimization and is based largely on the method outlined by 
Eisenstein et al. (ICML 2011), however we ignore the variance 
component (\tau) here.

Unfortunately, this implementation is rather tightly coupled to 
transcripts.py, in the sense that it relies on it for various counts
regarding the data, but it should be possible to modify
to handle other data sources. Moreover, we have not been able to 
secure IRB approval to release the actual data :(. 

Another note here is that this code (in addition to being rather coupled
to the task of patient-doctor communication) is extremely verbose; 
I have avoided all attempts to be clever, favoring explicitness. 
This means, however, that there is a lot of redundancy in the code.

Questions, &etc. should be sent to byron_wallace@brown.edu.
'''


import math
import copy
import pdb

import numpy
import scipy

import process_results

''' a few globals. '''
PRETTY_STR = "\n" + "".join(["-"]*30) + "\n"
THRESHOLD = .0001 # arbitrary but seems reasonable
ACC_THRESHOLD = .0025 # ditto 

### step size for descent step -- just make sure its small.
emission_gamma = transition_gamma = .025

class JointSequential:

    def __init__(self, tnb, joint=True, topics_only=False, transition_interactions=True):
        '''
        tnb -- this is a JointModel instance; its name ("tnb") is 
                due to obscure historical reasons ;). See the 
                transcripts.py module for its definition.
        '''
        self.tnb = tnb 
        self.iter = 0

        # this indicates whether or not we are using the 
        # joint -- topic and speech act -- model.
        # if not, then if topics_only is True, we assume that
        # are interested in modeling the topics (and topic
        # transitions)
        self.joint = joint 
        self.topics_only = topics_only # only matters if not joint

        # setup \eta vectors for each topic and speech act probabilities;
        # also setup the \sigma's for transitions
        self.topic_etas = {} 
        self.topic_proportions = {}

        self.speech_act_etas = {}
        self.speech_act_proportions = {}

        # emission pairs taking into consideration 
        # topic and speech act
        self.topic_sa_interaction_etas = {} 

        self.dimensions = tnb.get_dimensions()

        self.converged_topics = []
        self.converged_sas = []

        for topic in self.tnb.topic_set:
            self.topic_etas[topic] = numpy.zeros(self.dimensions)
            
        self.speech_act_etas = {}
        for sa in self.tnb.speech_act_set:
            self.speech_act_etas[sa] = numpy.zeros(self.dimensions)

        # for deltas in the likelihoods and accuracies 
        self.prev_ll = -float("inf")
        self.diff_ll = float("inf")
        self.prev_acc_y = -float("inf")
        self.prev_acc_s = -float("inf")
        self.acc_diff_y = float("inf")
        self.acc_diff_s = float("inf")
        self.diff_F = float("inf")
        self.prev_F = -float("inf")

        # the \Beta's reflect the 'adjusted' distributions
        # for the corresponding category.
        self.beta_topics = {}
        self.beta_sa = {}

        # pairs of topics/speech acts to adjusted transition
        # probabilities
        self.lambda_joint_y = {}
        self.lambda_joint_s = {}

        # componenents for transitions; analagous to \etas. 
        # the sigmas are the (exp) additive terms that perturb
        # the lambdas (transition probabilities)
        self.sigma_y_y = {} # topics to topics
        self.sigma_s_y = {} # speech acts to topics

        self.sigma_s_s = {} # speech acts to speech acts
        self.sigma_y_s = {} # topics to speech acts

        # interaction terms for *transitions*
        # these will map pairs to componenents
        self.sigma_interactions_y = {}
        self.sigma_interactions_s = {}
        self.transition_interactions = transition_interactions

        ### initialize sigmas
        for sa in self.tnb.get_speech_act_set(include_special_states=True):
            self.sigma_s_s[sa] = numpy.zeros(self.tnb.num_speech_acts)
            self.sigma_s_y[sa] = numpy.zeros(self.tnb.num_topics)
  
        for y in self.tnb.get_topic_set(include_special_states=True):
            self.sigma_y_y[y] = numpy.zeros(self.tnb.num_topics)
            self.sigma_y_s[y] = numpy.zeros(self.tnb.num_speech_acts)

        # we will only update pairs that occur more than
        # k times together
        pair_freq_min = 10
        self.frequent_pairs = []
        for pair in self.tnb.topic_sa_pairs:
            self.topic_sa_interaction_etas[pair] = numpy.zeros(self.dimensions)
            self.sigma_interactions_y[pair] = numpy.zeros(self.tnb.num_topics)
            self.sigma_interactions_s[pair] = numpy.zeros(self.tnb.num_speech_acts)

            if self.tnb.pair_counts[pair] > pair_freq_min and not "STOP" in pair and not "START" in pair:
                self.frequent_pairs.append(pair)
        print "modeling interactions for {0} pairs".format(len(self.frequent_pairs))

        # probabilities of words given topics, speech acts
        self.beta_joint = {}
        self.calc_beta_joint()

        self.calc_lambdas_joint()
       
        self.calc_topic_proportions()
        self.calc_speech_act_proportions()

    def calc_F_on_hold_out(self):
        '''
        calculates F-score on the hold out portion
        of the *training* dataset -- this is *not*
        looking at the test set. this is just an easy
        way to monitor performance / decide when to stop
        optimization.
        '''
        topic_preds, sa_preds = \
            self.predict_set_sequential_joint(self.tnb.held_out_cases_X)

        return process_results.calc_metrics(
                self.tnb.held_out_cases_Y, self.tnb.held_out_cases_S,
                preds_Y=topic_preds, preds_S=sa_preds)["avg_F"]

    def estimate_params(self, max_iters=100):
        print "calculating initial F ..."

        init_F = self.calc_F_on_hold_out()   
        print "initial F -- {0}".format(init_F)
        self.prev_F = init_F

        while self.diff_F > 0:
            self.step()
            cur_F = self.calc_F_on_hold_out()
            self.diff_F = cur_F - self.prev_F
            print "current F (on hold-out): {0}; previous F: {1}; diff: {2}".\
                format(cur_F, self.prev_F, self.diff_F)
            self.prev_F = cur_F
            self.iter += 1

    def step(self):
        ''' one optimization step. '''
        print "{0} on iteration {1}".format(PRETTY_STR, self.iter)

        # note that the betas are updated within this
        # method
        self.update_topic_etas(joint=True)
        print "ok. updating (speech act) etas..."
        self.update_speech_act_etas(joint=True) # again, betas are updated here
        print "now updating topic/sa **interaction** etas..."
        self.update_topic_sa_interaction_etas()
  
        ### assuming joint case here.
        print "updating (joint) transition lambdas..."
        self.update_transition_lambdas_joint()

        print "now updating topic/sa **interaction** sigmas"
        self.update_interaction_transition_lambdas_y()
        self.update_interaction_transition_lambdas_s()

        ll = self.joint_sequential_log_likelihood()


    def calc_topic_proportions(self):
        # note that we calc *log* probs
        N = float(len(self.tnb.Y))
        for topic in self.tnb.topic_set:
            p_topic = float(self.tnb.Y.count(topic)) / N
            print "p of topic {0}:{1}".format(topic, p_topic)
            self.topic_proportions[topic] = numpy.log(p_topic)

        print "(log) topic proportions calculated: {0}".\
            format(self.topic_proportions)

    def calc_speech_act_proportions(self):
        # again, we work on the log-scale
        N = float(len(self.tnb.S))
        for sa in self.tnb.speech_act_set:
            p_sa = float(self.tnb.S.count(sa)) / N
            print "p of speech act {0}:{1}".format(sa, p_sa)
            self.speech_act_proportions[sa] = numpy.log(p_sa)

        print "(log) speech act proportions calculated: {0}".\
            format(self.speech_act_proportions)

    def calc_lambdas(self):
        '''
        this is for the transitions in the *univariate* (not joint)
        case.  
        '''
        if self.topics_only:
            for y in self.tnb.get_topic_set(include_special_states=True):
                self.lambda_y_y[y] = self._calc_lambda(self.sigma_y_y[y], topic_trans=True)
        
        else:
            for s in self.tnb.get_speech_act_set(include_special_states=True):
                self.lambda_s_s[s] = self._calc_lambda(self.sigma_s_s[s], topic_trans=False)

    def _get_topic_trans_z(self, sigma_k):
        z = 0.0
        # j ranges over the number of topics.
        for j in xrange(self.tnb.num_topics):
            z += numpy.exp(sigma_k[j] + self.tnb.pi_topic[j])
        return z

    def _get_sa_trans_z(self, sigma_k):
        z = 0.0
        # and here j ranges over the number of speech acts.
        for j in xrange(self.tnb.num_speech_acts):
            z += numpy.exp(sigma_k[j] + self.tnb.pi_sa[j])
        return z
        

    def calc_lambdas_joint(self):
        for target_y in self.tnb.get_topic_set(include_special_states=True):    
            for pair in self.tnb.topic_sa_pairs:
                self.lambda_joint_y[pair] = self._calc_lambda_joint(pair, to_topic_trans=True)
     
        for target_sa in self.tnb.get_speech_act_set(include_special_states=True):    
            for pair in self.tnb.topic_sa_pairs:
                self.lambda_joint_s[pair] = self._calc_lambda_joint(pair, to_topic_trans=False)


    def _calc_lambda_joint(self, pair, to_topic_trans=True):
        topic, sa = pair
        # \pi is the (log) background frequency
        pi = self.tnb.pi_topic if to_topic_trans else self.tnb.pi_sa

        # both of these are vectors of length |topics|
        sigma_y, sigma_s = None, None
        sigma_interaction = None
        if to_topic_trans:
            sigma_y = self.sigma_y_y[topic]
            sigma_s = self.sigma_s_y[sa]
            sigma_interaction = self.sigma_interactions_y[pair]
        else:
            sigma_y = self.sigma_y_s[topic]
            sigma_s = self.sigma_s_s[sa]
            sigma_interaction = self.sigma_interactions_s[pair]           

        lambda_joint = numpy.exp(pi + sigma_y + sigma_s + sigma_interaction)
        z = sum(lambda_joint)
        return lambda_joint/z

    def _calc_lambda(self, sigma_k, topic_trans=True):
        '''
            \lambda_k <- exp(\sigma_k + \pi) / \sum_j(exp(\sigma_kj + \pi_i))

            note that \pi will be either \pi_topic or \pi_sa, depending on which
            we are updating a component of
        '''
        # \pi is the (log) background frequency
        pi = self.tnb.pi_topic if topic_trans else self.tnb.pi_sa

        lambda_k = numpy.exp(sigma_k + pi)
        # normalize

        z = self._get_topic_trans_z(sigma_k) if topic_trans else self._get_sa_trans_z(sigma_k)
        if z == 0:
            # something is wrong -- should never happen
            pdb.set_trace()
        return lambda_k/z

    def calc_beta_joint(self):
        '''
        word distribution adjusted for both topics and 
        speech acts
        '''
        for topic in self.tnb.topic_set:
            topic_eta = self.topic_etas[topic]

            for sa in self.tnb.speech_act_set:
                sa_eta = self.speech_act_etas[sa]
                ### add the interaction term
                interaction_eta = self.topic_sa_interaction_etas[(topic, sa)]
                beta_t_sa = self._calc_beta(topic_eta + sa_eta + interaction_eta)

                self.beta_joint[(topic, sa)] = beta_t_sa

    def y_transition_prob_joint(self, y, y_prev, s_prev, z=None):
        '''
        probability of transitioning to y given that the previous
        topic was y_prev and the previous speech act was s_prev
        '''
        if z is None:
            z = self.z_for_joint_to_y(y_prev, s_prev)

        y_index = self.tnb.topics_to_indices[y]
        pi_y = self.tnb.pi_topic[y_index]

        trans_prob = numpy.exp(\
                        pi_y +\
                        self.sigma_y_y[y_prev][y_index] +\
                        self.sigma_s_y[s_prev][y_index] +\
                        self.sigma_interactions_y[(y_prev, s_prev)][y_index])
        trans_prob = trans_prob/z

        return trans_prob

    def s_transition_prob_joint(self, s, y_prev, s_prev, z=None):
        if z is None:
            z = self.z_for_joint_to_s(y_prev, s_prev)

        s_index = self.tnb.speech_acts_to_indices[s]
        pi_s = self.tnb.pi_sa[s_index]

        trans_prob = numpy.exp(\
                        pi_s +\
                        self.sigma_s_s[s_prev][s_index] +\
                        self.sigma_y_s[y_prev][s_index] +\
                        self.sigma_interactions_s[(y_prev, s_prev)][s_index])
        trans_prob = trans_prob/z

        return trans_prob

    def _calc_beta(self, eta_k):
        '''
            \beta_k <- exp(\eta_k + m) / \sum_i(exp(\eta_ki + m_i))
        '''
        beta_k = numpy.exp(eta_k + self.tnb.m)
        
        # renormalize
        z = 0.0
        for i in xrange(self.dimensions):
            z += numpy.exp(eta_k[i] + self.tnb.m[i])

        # for now we will leave vectors as 1xw, but note that
        # later we will assume that beta's are wx1 vectors, 
        # rather than 1xw; hence you will need to transpose these, e.g.,
        #   > numpy.mat(beta_k).T
        return beta_k / z

    ''' univariate '''
    def calc_topic_betas(self):
        for topic in self.tnb.topic_set:
            self.beta_topics[topic] = self._calc_beta(self.topic_etas[topic])

    def calc_speech_act_betas(self):
        for speech_act in self.tnb.speech_act_set:
            self.beta_sa[speech_act] = self._calc_beta(self.speech_act_etas[speech_act])

    def update_topic_sa_interaction_etas(self):
        print "updating ({0}) pairs".format(len(self.frequent_pairs))

        prev_ll = self.joint_sequential_log_likelihood()
        for pair in self.frequent_pairs:
            delta = self.get_delta_interaction_eta(pair)
            delta = numpy.array(delta.T)[0] # this is (w,)
            prev_interaction_eta = copy.copy(self.topic_sa_interaction_etas[pair])
            self.topic_sa_interaction_etas[pair] = \
                    self.topic_sa_interaction_etas[pair] - (emission_gamma*delta)

            print "\nupdating interaction pair: {0}".format(pair)
            print "pair {0} delta: {1}".format(pair, delta)
            print "eta_interaction: {0}\n".format(self.topic_sa_interaction_etas[pair])

            self.calc_beta_joint()

            ll = self.joint_sequential_log_likelihood()
            print "ll after updating interaction {0}:{1}".format(pair, ll)
            diff_ll_for_pair = ll - prev_ll 
            print "diff ll {0}".format(diff_ll_for_pair)

            if diff_ll_for_pair < THRESHOLD or math.isnan(ll):
                # 'reject'
                print "updated rejected!"
                self.topic_sa_interaction_etas[pair] = prev_interaction_eta
            else:
                prev_ll = ll

        self.calc_beta_joint()     

    def update_speech_act_etas(self, joint=True):
        '''
        update speech act component \eta_k according to:

            \eta_k^t <- \eta_k^(t-1) - \delta \eta_k
        '''
        if joint:
            prev_ll = self.joint_sequential_log_likelihood()
        else:
            prev_ll = self.speech_act_log_likelihood()

        speech_acts = [sa for sa in self.tnb.speech_act_set if not sa in self.converged_sas]
        for sa in speech_acts:
            delta = None
            if joint:
                delta = self.get_delta_speech_act_joint(sa)
            else:
                delta = self.get_delta_speech_act(sa)

            delta = numpy.array(delta.T)[0] # this is (w,)
            prev_etas_for_sa = copy.copy(self.speech_act_etas[sa])
            # update
            self.speech_act_etas[sa] = self.speech_act_etas[sa] - (emission_gamma*delta)

            print "\n\n\nupdating speech act: {0}".format(sa)
            print "speech act {0} delta: {1}".format(sa, delta)
            print "eta_sa: {0}\n".format(self.speech_act_etas[sa])
            
            ll = None
            if joint:
                # note that joint_sequential_log_likelihood does not
                # use the betas directly, but rather uses the \etas.
                ll = self.joint_sequential_log_likelihood()
            else:
                self.calc_speech_act_betas()
                ll = self.speech_act_log_likelihood()

            print "ll after updating speech act {0}:{1}".format(sa, ll)
            diff_ll_for_sa = ll - prev_ll 
            print "diff ll {0}".format(diff_ll_for_sa)

            ## if the likelihood decreases (or increases negligibly)
            if diff_ll_for_sa < THRESHOLD or math.isnan(ll):
                print "speech act {0} has converged!".format(sa)
                self.speech_act_etas[sa] = prev_etas_for_sa
                # note that we don't update the prev_ll here.
            else:
                prev_ll = ll
            
        # update the \betas to account for new
        # speech act \etas
        if joint:
            self.calc_beta_joint()
        else:
            self.calc_speech_act_betas()

    def _contains_any(self, l, x):
        '''
        helper method; return True iff any elements of x are in l
        '''
        for x_i in x:
            if x_i in l:
                return True
        return False

    def _pairs_without_special_states(self, include_start=False):
        to_remove = ["STOP"]
        if not include_start:
            to_remove.append("START")

        return [pair for pair in self.tnb.topic_sa_pairs \
                            if not self._contains_any(pair, to_remove)]

    def calc_joint_Zs(self):
        joint_Zs = {}
        #for y,s in self.tnb.topic_sa_pairs:
        for y,s in self._pairs_without_special_states():
            joint_Zs[(y,s)] = self.get_joint_z(y, s)
        return joint_Zs

    def get_joint_z(self, y, s):
        return sum(numpy.exp(self.tnb.m + \
                    self.topic_etas[y] + \
                    self.speech_act_etas[s] +\
                    self.topic_sa_interaction_etas[(y,s)]))

    def g_eta_topic_joint(self, topic):
        '''
        partial derivative for the component corresponding to the
        given topic, taking into consideration the joint \betas.
        '''
        observed_topic_counts = self.tnb.get_c_topic(topic)

        expected = numpy.zeros(self.tnb.get_dimensions())
        # iterate (/marginalize) over speech acts
        for sa in self.tnb.speech_act_set:
            # observed word counts for this topic and speech act pair
            c_topic_sa = self.tnb.get_c_joint(topic, sa)
            C_topic_sa = sum(c_topic_sa)
            
            # joint beta reflecting word distribution for this topic/speech
            # act pair
            beta_t_sa = self.beta_joint[(topic, sa)] 
            # expected word counts
            expected += C_topic_sa * beta_t_sa

        partial_deriv = observed_topic_counts - expected
        partial_deriv = numpy.matrix(partial_deriv).T
        return partial_deriv


    def g_eta_speech_act_joint(self, speech_act):
        '''
        partial derivative for the component corresponding to the
        given speech act, taking into consideration the joint \betas 
        (i.e., taking into account the topics).
        '''
        observed_speech_act_word_counts = self.tnb.get_c_speech_act(speech_act)
        expected = numpy.zeros(self.tnb.get_dimensions())
       
        # iterate over topics
        for topic in self.tnb.topic_set:
            # observed word counts for this topic and speech act pair
            c_topic_sa = self.tnb.get_c_joint(topic, speech_act)
            C_topic_sa = sum(c_topic_sa)

            # joint beta reflecting word distribution for this topic/speech
            # act pair
            beta_t_sa = self.beta_joint[(topic, speech_act)] 
            expected += C_topic_sa * beta_t_sa
        
        partial_deriv = observed_speech_act_word_counts - expected
        partial_deriv = numpy.matrix(partial_deriv).T
        return partial_deriv

    def g_eta_interaction(self, pair):
        '''
        partial derivative for the component corresponding to the
        given *interaction pair* (topic, speech act).
        '''
        #observed_pair_word_counts = self.tnb.joint_token_counts[pair]
        observed_pair_word_counts = self.tnb.get_c_pair(pair)
        # the expected is just the number of times we observed this
        # (topic, speech act) pair times the current beta
        num_times_pair_observed = sum(observed_pair_word_counts) #self.tnb.pair_counts[pair]
        expected = num_times_pair_observed * self.beta_joint[pair]

        partial_deriv = observed_pair_word_counts - expected
        partial_deriv = numpy.matrix(partial_deriv).T
        return partial_deriv

    def update_topic_etas(self, joint=False):
        '''
        update topic eta according to:

            \eta_k^t <- \eta_k^(t-1) - \delta \eta_k

        if joint is True, then we take the speech act
        etas into consideration when we update the topic_etas 
        '''
        prev_ll = None
        if joint:
            prev_ll = self.joint_sequential_log_likelihood()
        else:
            prev_ll = self.topic_log_likelihood()

        topics = [t for t in self.tnb.topic_set if not t in self.converged_topics]
        for topic in topics:
            print "getting delta for topic {0}".format(topic)
            delta = None
            if joint:
                delta = self.get_delta_topic_joint(topic)
            else:
                delta = self.get_delta_topic(topic)
           
            # you have to transform the delta 'matrix' here
            # into an array, otherwise you get 'matrix too big' 
            # exceptions.
            delta = numpy.array(delta.T)[0] # this is (w,)
            prev_etas_for_topic = copy.copy(self.topic_etas[topic])
            # update
            self.topic_etas[topic] = self.topic_etas[topic] - (emission_gamma*delta)

            print "\nupdating topic: {0}".format(topic)
            print "topic {0} delta: {1}".format(topic, delta)
            print "eta_topic: {0}\n".format(self.topic_etas[topic])
            
            ll = None
            if joint:
                ll = self.joint_sequential_log_likelihood()
            else:
                ll = self.topic_log_likelihood()
            print "ll after updating topic {0}:{1}".format(topic, ll)
            diff_ll_for_topic = ll - prev_ll 
            print "diff in ll: {0}".format(diff_ll_for_topic)

            # @TODO raise exception on isnan -- or at least a warning --
            # because this indicates badness (probably)
            if diff_ll_for_topic < THRESHOLD or math.isnan(ll):
                print "topic {0} has converged!".format(topic)
                # use previous value
                self.topic_etas[topic] = prev_etas_for_topic
                self.converged_topics.append(topic)
            else:
                prev_ll = ll 
        
        # update betas to reflect new topic \etas
        if joint:
            self.calc_beta_joint()
        else:
            self.calc_topic_betas()


    def update_transition_lambdas_joint(self):
        topics_to_update = self.tnb.get_topic_set(include_special_states=True, exclude_stop_state=True)
        speech_acts_to_update = self.tnb.get_speech_act_set(include_special_states=True, exclude_stop_state=True)

        # catch the previous ll
        prev_ll = self.joint_sequential_log_likelihood()
        before_lambda_updates_ll = prev_ll

        print "updating y->y transition sigmas"
        ''' first update the topic to topic transitions and the speech act
                to speech act transitions '''
        for topic in topics_to_update:
            # old value
            old_sigma = self.sigma_y_y[topic]
            # update topic-to-topic transition components
            delta_topic_y = self.get_delta_y_y_joint(topic)
            delta_topic_y = numpy.array(delta_topic_y.T)[0]
            self.sigma_y_y[topic] = self.sigma_y_y[topic] - (transition_gamma*delta_topic_y)
            new_ll = self.joint_sequential_log_likelihood()
            print "ll - prev_ll: {0}".format(new_ll-prev_ll)
            if (new_ll - prev_ll) < THRESHOLD:
                # then don't update this sigma
                print "not updating y->y for topic {0}".format(topic)
                self.sigma_y_y[topic] = old_sigma
            else:
                prev_ll = new_ll


        prev_ll = self.joint_sequential_log_likelihood()
        print "\nupdating s->s transition sigmas"
        for speech_act in speech_acts_to_update:
            old_sigma = self.sigma_s_s[speech_act]
            delta_sa_s = self.get_delta_s_s_joint(speech_act)
            delta_sa_s = numpy.array(delta_sa_s.T)[0]
            self.sigma_s_s[speech_act] = self.sigma_s_s[speech_act] - (transition_gamma*delta_sa_s)
            #print "speech act delta = {0} (update: {1})".format(delta_sa_s, transition_gamma*delta_sa_s)
            new_ll = self.joint_sequential_log_likelihood()
            print "ll - prev_ll: {0}".format(new_ll-prev_ll)
            if (new_ll - prev_ll) < THRESHOLD:
                # then don't update this sigma
                print "not updating s->s for speech act {0}".format(speech_act)
                self.sigma_s_s[speech_act] = old_sigma
            else:
                prev_ll = new_ll

        # recalculate lambdas
        self.calc_lambdas_joint()
        print "\n\nlikelihood after updating 'primary' transitions: {0} (difference of {1})".\
                format(new_ll, new_ll-before_lambda_updates_ll)
        ll_after_primary = new_ll

        print "updating y->s transition sigmas"
        ''' now update the 'secondary' transitions: topics to speech acts; speech 
                acts to topics '''
        for topic in topics_to_update:
            old_sigma = self.sigma_y_s[topic]
            delta_topic_s = self.get_delta_y_s_joint(topic)
            # fixes the dimensions to be (|speech acts|,)
            delta_topic_s = numpy.array(delta_topic_s.T)[0]
            self.sigma_y_s[topic] = self.sigma_y_s[topic] - (transition_gamma*delta_topic_s)
            new_ll = self.joint_sequential_log_likelihood()
            print "ll - prev_ll: {0}".format(new_ll-prev_ll)
            if (new_ll - prev_ll) < THRESHOLD:
                # then don't update this sigma
                print "not updating y->s for topic {0}".format(topic)
                self.sigma_y_s[topic] = old_sigma
            else:
                prev_ll = new_ll

        self.calc_lambdas_joint()

        print "updating s->y transition sigmas"
        for speech_act in speech_acts_to_update:
            old_sigma = self.sigma_s_y[speech_act]
            delta_sa_y = self.get_delta_s_y_joint(speech_act) 
            delta_sa_y = numpy.array(delta_sa_y.T)[0]
            self.sigma_s_y[speech_act] = self.sigma_s_y[speech_act] - (transition_gamma*delta_sa_y)
            new_ll = self.joint_sequential_log_likelihood()
            print "ll - prev_ll: {0}".format(new_ll-prev_ll)
            if (new_ll - prev_ll) < THRESHOLD:
                # then don't update this sigma
                print "not updating s->y for speech act {0}".format(speech_act)
                self.sigma_s_y[speech_act] = old_sigma
            else:
                prev_ll = new_ll


        print "likelihood after updating 'secondary' transitions: {0} (difference of {1})".\
                format(new_ll, new_ll-ll_after_primary)

        self.calc_lambdas_joint()

    def update_interaction_transition_lambdas_y(self):
        prior_ll = prev_ll = self.joint_sequential_log_likelihood()
        print "ll prior to updating interaction transitions (y): {0}".format(prior_ll)
        for pair in self.frequent_pairs:
            delta_interaction_y = self.get_delta_interaction_y(pair)
            delta = numpy.array(delta_interaction_y.T)[0] 
            prev_val = copy.deepcopy(self.sigma_interactions_y[pair])
            self.sigma_interactions_y[pair] = self.sigma_interactions_y[pair] - (transition_gamma*delta)
            new_ll = self.joint_sequential_log_likelihood()
            print "previous ll: {0}, new ll {1}, diff: {2}".format(prev_ll, new_ll, new_ll - prev_ll)
            if new_ll - prev_ll < THRESHOLD:
                print "rejecting interaction transition update!"
                self.sigma_interactions_y[pair] = prev_val
            else:
                print "new sigma interactions (y) for pair {0}: {1}".format(pair, self.sigma_interactions_y[pair])
                prev_ll = new_ll

        print "delta ll after updating interaction transition terms for topics: {0}".\
            format(prev_ll - prior_ll)
        self.calc_lambdas_joint()

    ### interactions
    def get_delta_interaction_y(self, pair):
        topic, sa = pair
        K_pair_y = self.calc_k_pair_y(pair) # (diagonal) |topics|x|topics|
        g_pair_y = self.g_sigma_pair_y(pair) 

        K_gradiant = K_pair_y * g_pair_y # |topics x 1|
        T_pair_y = self.tnb.get_T_y_joint(topic, sa) # scalar

        lambda_pair_y = numpy.mat(self.lambda_joint_y[pair]).T # |topics| x 1
        TKL = T_pair_y * K_pair_y * lambda_pair_y # |topics|x1

        z = 1 + T_pair_y * lambda_pair_y.T * K_pair_y * lambda_pair_y
        if z == 0:
            print " delta interaction y -- divisor is 0... setting to 1."
            z = 1.0

        delta_pair_y = K_gradiant - TKL/z * (lambda_pair_y.T * K_gradiant)
        return delta_pair_y

    def update_interaction_transition_lambdas_s(self):
        prior_ll = prev_ll = self.joint_sequential_log_likelihood()
        for pair in self.frequent_pairs:
            delta_interaction_s = self.get_delta_interaction_s(pair)
            delta = numpy.array(delta_interaction_s.T)[0] 
            prev_val = copy.copy(self.sigma_interactions_s[pair])
            self.sigma_interactions_s[pair] = self.sigma_interactions_s[pair] - (transition_gamma*delta)
            new_ll = self.joint_sequential_log_likelihood()
            print "previous ll: {0}, new ll {1}, diff: {2}".format(prev_ll, new_ll, new_ll - prev_ll)
            if new_ll - prev_ll < THRESHOLD:
                print "rejecting interaction transition update!"
                self.sigma_interactions_s[pair] = prev_val
            else:
                print "new sigma interactions (s) for pair {0}: {1}".format(pair, self.sigma_interactions_s[pair])
                prev_ll = new_ll
        print "delta ll after updating interaction transition terms for speech acts: {0}".\
            format(prev_ll - prior_ll)
        self.calc_lambdas_joint()

    def get_delta_interaction_s(self, pair):
        topic, sa = pair
        K_pair_s = self.calc_k_pair_s(pair) # (diagonal) |speech acts|x|speech acts|
        g_pair_s = self.g_sigma_pair_s(pair) 

        K_gradiant = K_pair_s * g_pair_s # |speech acts x 1|
        T_pair_s = self.tnb.get_T_s_joint(topic, sa) # scalar

        lambda_pair_s = numpy.mat(self.lambda_joint_s[pair]).T # |speech acts| x 1
        TKL = T_pair_s * K_pair_s * lambda_pair_s # |speech acts|x1
        z = 1 + T_pair_s * lambda_pair_s.T * K_pair_s * lambda_pair_s
        if z == 0:
            print " delta interaction s -- divisor is 0... setting to 1."
            z = 1.0

        delta_pair_s = K_gradiant - TKL/z * (lambda_pair_s.T * K_gradiant)

        return delta_pair_s

    '''
    these delta's are for the emission probabilities
    '''
    def get_delta_topic_joint(self, topic):
        g_eta_topic = self.g_eta_topic_joint(topic) # wx1
        H_inv_topic_joint = self.get_H_inv_topic_joint(topic) # wxw
        return H_inv_topic_joint * g_eta_topic

    def get_delta_speech_act_joint(self, topic):
        g_eta_sa = self.g_eta_speech_act_joint(topic) # wx1
        H_inv_speech_act_joint = self.get_H_inv_speech_act_joint(topic) # wxw
        return H_inv_speech_act_joint * g_eta_sa

    def _invert_v(self, v):
        # v cannot contain any zeros!
        return [1.0/v_i for v_i in v]

    def get_A_topic_joint(self, topic):
        A_topic = numpy.zeros(self.tnb.get_dimensions())
        for sa in self.tnb.get_speech_act_set():
            C_topic_sa = self.tnb.get_C_joint(topic, sa)
            beta_topic_sa = self.beta_joint[(topic, sa)]
            A_topic += C_topic_sa * beta_topic_sa

        A_topic = self._invert_v(-1 * A_topic)
        m = n = self.tnb.get_dimensions()
        return scipy.sparse.spdiags(A_topic, 0, m, n)

    def get_A_speech_act_joint(self, sa):
        A_sa = numpy.zeros(self.tnb.get_dimensions())
        # take the expectation over topics
        for topic in self.tnb.get_topic_set():
            C_topic_sa = self.tnb.get_C_joint(topic, sa)
            beta_topic_sa = self.beta_joint[(topic, sa)]
            A_sa += C_topic_sa * beta_topic_sa

        A_sa = self._invert_v(-1 * A_sa)
        m = n = self.tnb.get_dimensions()
        return scipy.sparse.spdiags(A_sa, 0, m, n)


    def get_H_inv_topic_joint(self, topic):
        m = self.tnb.get_dimensions()
        H_inv = numpy.zeros((m,m))
        A = self.get_A_topic_joint(topic)
        for sa in self.tnb.get_speech_act_set(include_special_states=False):
            C_topic_sa = self.tnb.get_C_joint(topic, sa)
            beta_topic_sa = self.beta_joint[(topic, sa)]
            numerator = \
                A * C_topic_sa * beta_topic_sa * beta_topic_sa.T * A

            denom = \
                1 + C_topic_sa * beta_topic_sa.T * A * beta_topic_sa
                
            H_inv += numerator / denom
       
        H_inv = A - H_inv 
        return H_inv

    def get_H_inv_speech_act_joint(self, speech_act):
        m = self.tnb.get_dimensions()
        H_inv = numpy.zeros((m,m))
        A = self.get_A_speech_act_joint(speech_act)
        for topic in self.tnb.get_topic_set(include_special_states=False):
            C_topic_sa = self.tnb.get_C_joint(topic, speech_act)
            beta_topic_sa = self.beta_joint[(topic, speech_act)]
            numerator = \
                A * C_topic_sa * beta_topic_sa * beta_topic_sa.T * A
            denom = \
                1 + C_topic_sa * beta_topic_sa.T * A * beta_topic_sa
            H_inv += numerator / denom
       
        H_inv = A - H_inv 
        return H_inv

    def get_A_interaction(self, pair):
        C_pair = sum(self.tnb.get_c_pair(pair))#self.tnb.pair_counts[pair]
        beta_topic_sa = self.beta_joint[pair]
        A_interaction = -1* (C_pair * beta_topic_sa)
        A_interaction = self._invert_v(A_interaction)
        m = n = self.tnb.get_dimensions()
        return scipy.sparse.spdiags(A_interaction, 0, m, n)

    def get_delta_interaction_eta(self, pair):
        m = self.tnb.get_dimensions()
        H_inv = numpy.zeros((m,m))
        A = self.get_A_interaction(pair) # wxw
        g_eta_interaction = self.g_eta_interaction(pair) # wx1
        A_gradiant = A * g_eta_interaction # wx1
   
        C_pair = sum(self.tnb.get_c_pair(pair))#self.tnb.pair_counts[pair] # scalar (1x1)
        beta_pair = numpy.mat(self.beta_joint[pair]).T # wx1
        
        CAB = C_pair * A * beta_pair # numerator -- wx1
        z = 1 + C_pair * beta_pair.T * A * beta_pair # the divisor; a scalar (1x1)
        if z == 0:
            z = 1.0

        delta_pair = A_gradiant - CAB/z * (beta_pair.T * A_gradiant) # ultimately wx1
        
        return delta_pair

    def calc_k_pair_y(self, pair): # from pairs to topics
        topic, sa = pair
        T_pair_y = self.tnb.get_T_y_joint(topic, sa)
        lambda_pair_y = self.lambda_joint_y[pair]
        k_pair_y = -1 * (T_pair_y * lambda_pair_y) # expected
        k_pair_y = self._invert_v(k_pair_y)
        m = n = self.tnb.num_topics
        return scipy.sparse.spdiags(k_pair_y, 0, m, n)

    def calc_k_pair_s(self, pair): # from pairs to speech acts
        topic, sa = pair
        T_pair_s = self.tnb.get_T_s_joint(topic, sa)
        lambda_pair_s = self.lambda_joint_s[pair]
        k_pair_s = -1 * (T_pair_s * lambda_pair_s) # expected
        k_pair_s = self._invert_v(k_pair_s)
        m = n = self.tnb.num_speech_acts
        return scipy.sparse.spdiags(k_pair_s, 0, m, n)

    '''
    these deltas are for the transition probabilities.
    '''
    def get_delta_y_y_joint(self, topic):
        g_y_y_joint = self.g_sigma_y_y_joint(topic)
        H_y_y_joint_inv = self.get_H_inv_transition_y_y_joint(topic)
        return H_y_y_joint_inv * g_y_y_joint

    def get_delta_y_s_joint(self, topic):
        g_y_s_joint = self.g_sigma_y_s_joint(topic)
        H_y_s_joint_inv = self.get_H_inv_transition_y_s_joint(topic)
        return H_y_s_joint_inv * g_y_s_joint

    def get_delta_s_s_joint(self, sa):
        g_s_s_joint = self.g_sigma_s_s_joint(sa)
        H_s_s_joint_inv = self.get_H_inv_transition_s_s_joint(sa)
        return H_s_s_joint_inv * g_s_s_joint

    def get_delta_s_y_joint(self, sa):
        g_s_y_joint = self.g_sigma_s_y_joint(sa)
        H_s_y_joint_inv = self.get_H_inv_transition_s_y_joint(sa)
        return H_s_y_joint_inv * g_s_y_joint

    def g_sigma_y_y_joint(self, topic):
        # first get the observed transition counts out of
        # topic topic
        # 1 x |topics|. observed transition counts out of topic 
        # (into other topics)
        t_y_y = self.tnb.get_t_y_y(topic)   
        T_y_y = sum(t_y_y) # scalar (sum); total transitions out of topic
        # now calculate the expected, marginalizing over
        # the speech acts
        expected = 0.0
        for sa in self.tnb.get_speech_act_set(include_special_states=True):
            pair = (topic, sa)
            # expected transition probabilities, under current model
            cur_lambda_joint = self.lambda_joint_y[pair]
            C_topic_sa = self.tnb.get_T_y_joint(topic, sa)
            expected += cur_lambda_joint * C_topic_sa
        partial_deriv =  t_y_y - expected
        partial_deriv = numpy.matrix(partial_deriv).T
        return partial_deriv

    def g_sigma_y_s_joint(self, topic):
        t_y_s = self.tnb.get_t_y_s(topic)   # 1 x |speech acts|. 
        T_y_s = sum(t_y_s) # scalar (sum); total transitions out of topic into speech acts
        # now calculate the expected, 'marginalizing' over speech acts
        expected = 0.0
        for sa in self.tnb.get_speech_act_set(include_special_states=True):
            pair = (topic, sa)
            # expected transition probabilities, under current model
            cur_lambda_joint = self.lambda_joint_s[pair]
            # total number of times we were in this topic *and*
            # this speech act
            C_topic_sa = self.tnb.get_T_s_joint(topic, sa)
            expected += cur_lambda_joint * C_topic_sa
        partial_deriv =  t_y_s - expected
        partial_deriv = numpy.matrix(partial_deriv).T
        return partial_deriv

    def g_sigma_s_y_joint(self, speech_act):
        t_s_y = self.tnb.get_t_s_y(speech_act)   # 1 x |topics|. observed transition counts from speech act to topics.
        T_s_y = sum(t_s_y) # scalar (sum); total transitions out of speech act
        # now calculate the expected, marginalizing over topics
        expected = 0.0
        for topic in self.tnb.get_topic_set(include_special_states=True):
            pair = (topic, speech_act)
            # expected transition probabilities, under current model
            cur_lambda_joint = self.lambda_joint_y[pair]
            # total number of times we were in this topic *and* speech act
            C_topic_sa = self.tnb.get_T_y_joint(topic, speech_act)
            expected += cur_lambda_joint * C_topic_sa
        partial_deriv =  t_s_y - expected
        partial_deriv = numpy.matrix(partial_deriv).T
        return partial_deriv

    def g_sigma_s_s_joint(self, speech_act):
        t_s_s = self.tnb.get_t_s_s(speech_act)   # 1 x |topics|. observed transition counts from speech act to speech acts
        T_s_s = sum(t_s_s) # scalar (sum); total transitions from speech act to speech acts
        # now calculate the expected, marginalizing over topics
        expected = 0.0
        for topic in self.tnb.get_topic_set(include_special_states=True):
            pair = (topic, speech_act)
            # expected transition probabilities, under current model
            cur_lambda_joint = self.lambda_joint_s[pair]
            # total number of times we were in this topic *and* speech act
            C_topic_sa = self.tnb.get_T_s_joint(topic, speech_act)
            expected += cur_lambda_joint * C_topic_sa
        partial_deriv =  t_s_s - expected
        partial_deriv = numpy.matrix(partial_deriv).T
        return partial_deriv

    '''
    partial derivative for interaction effect of topics x speech 
    acts -> topics
    '''
    def g_sigma_pair_y(self, pair):
        topic, sa = pair 
        # observed counts
        t_pair_y = self.tnb.get_t_y_joint(topic, sa) # 1 x |topics|
        T_pair_y = sum(t_pair_y) # total transitions out of topic, sa
        lambda_pair_y = self.lambda_joint_y[pair]
        # observed minus expected
        partial_deriv = t_pair_y - (T_pair_y * lambda_pair_y)
        partial_deriv = numpy.matrix(partial_deriv).T
        return partial_deriv

    '''
    partial derivative for interaction effect of topics x speech 
    acts -> speech acts
    '''
    def g_sigma_pair_s(self, pair):
        topic, sa = pair 
        # observed counts
        t_pair_s = self.tnb.get_t_s_joint(topic, sa) # 1 x |topics|
        T_pair_s = sum(t_pair_s) # total transitions out of topic, sa
        lambda_pair_s = self.lambda_joint_s[pair]
        # observed minus expected
        partial_deriv = t_pair_s - (T_pair_s * lambda_pair_s)
        partial_deriv = numpy.matrix(partial_deriv).T
        return partial_deriv

    '''
    The 'A' helper matries for the (joint) transition
    probabilities out of topic/speech act pairs and
    into topics.
    '''
    def get_A_s_y_joint(self, sa):
        A_s_y = numpy.zeros(self.tnb.num_topics)
        for topic in self.tnb.get_topic_set(include_special_states=True):
            T_s_y = self.tnb.get_T_y_joint(topic, sa)
            # current joint transition probabilities *into* topics
            lambda_joint_y = self.lambda_joint_y[(topic, sa)] 
            A_s_y += T_s_y * lambda_joint_y # expected transition counts (under model)

        A_s_y = self._invert_v(-1*A_s_y)
        m = n = self.tnb.num_topics
        return scipy.sparse.spdiags(A_s_y, 0, m, n)

    def get_A_y_y_joint(self, topic):
        A_y_y = numpy.zeros(self.tnb.num_topics)
        for sa in self.tnb.get_speech_act_set(include_special_states=True):
            # number of times we've transitioned out of this pair
            T_y_s = self.tnb.get_T_y_joint(topic, sa) 
            # current (adjusted) transition probabilities into topics
            lambda_joint_y = self.lambda_joint_y[(topic, sa)]
            A_y_y += T_y_s * lambda_joint_y # expected transition counts (under model)

        # inverse
        A_y_y = self._invert_v(-1*A_y_y)
        m = n = self.tnb.num_topics
        return scipy.sparse.spdiags(A_y_y, 0, m, n)


    '''
    The 'A' helper matries for the (joint) transition
    probabilities out of topic/speech act pairs and
    into speech acts.
    '''
    def get_A_s_s_joint(self, sa): 
        ''' speech act -> speech act (adjusted for topics) '''
        A_s_s = numpy.zeros(self.tnb.num_speech_acts) # oh please, grow up ;)
        for topic in self.tnb.get_topic_set(include_special_states=True):
            T_s_y = self.tnb.get_T_s_joint(topic, sa)
            # current transition probabilities into *speech acts*
            lambda_joint_s = self.lambda_joint_s[(topic, sa)]
            A_s_s += T_s_y * lambda_joint_s # expected transition counts

        A_s_s = self._invert_v(-1 * A_s_s)
        m = n = self.tnb.num_speech_acts
        return scipy.sparse.spdiags(A_s_s, 0, m, n)

    def get_A_y_s_joint(self, topic):
        ''' topic -> speech act (adjusted for speech acts) '''
        A_y_s = numpy.zeros(self.tnb.num_speech_acts)
        for sa in self.tnb.get_speech_act_set(include_special_states=True):
            T_y_s = self.tnb.get_T_y_joint(topic, sa)
            lambda_joint_s = self.lambda_joint_s[(topic, sa)]
            A_y_s += T_y_s * lambda_joint_s # expected transition counts

        A_y_s = -1 * A_y_s
        A_y_s = self._invert_v(A_y_s)
        m = n = self.tnb.num_speech_acts
        return scipy.sparse.spdiags(A_y_s, 0, m, n)

    def get_H_inv_transition_y_y_joint(self, topic):
        #  topic to topics
        m = self.tnb.num_topics
        H_inv = numpy.zeros((m, m))
        A_transition = self.get_A_y_y_joint(topic)
        # sum over speech acts
        for sa in self.tnb.get_speech_act_set(include_special_states=True):
            pair = (topic, sa)
            # observed number of transitions out of this topic/sa pair
            T_s_y = self.tnb.get_T_y_joint(topic, sa)
            
            # expected transition probabilities, under current model
            cur_lambda_joint = self.lambda_joint_y[pair] 

            numerator = A_transition * T_s_y * cur_lambda_joint * cur_lambda_joint.T * A_transition

            denom = 1 + T_s_y * cur_lambda_joint.T * A_transition * cur_lambda_joint

            H_inv += numerator / denom

        H_inv = A_transition - H_inv
        return H_inv

    def get_H_inv_transition_s_y_joint(self, sa):
        # speech act to topics; the only difference between this
        # and above is that we sum over topics here, holding the 
        # speech act constant.
        m = self.tnb.num_topics
        H_inv = numpy.zeros((m, m))
        A_transition = self.get_A_s_y_joint(sa)
        for topic in self.tnb.get_topic_set(include_special_states=True):
            pair = (topic, sa)
            T_s_y = self.tnb.get_T_y_joint(topic, sa)
            cur_lambda_joint = self.lambda_joint_y[pair]
            numerator = A_transition * T_s_y * cur_lambda_joint * cur_lambda_joint.T * A_transition
            denom = 1 + T_s_y * cur_lambda_joint.T * A_transition * cur_lambda_joint
            H_inv += numerator / denom
        H_inv = A_transition - H_inv
        return H_inv

    def get_H_inv_transition_s_s_joint(self, sa):
        # speech acts to speech acts
        m = self.tnb.num_speech_acts
        H_inv = numpy.zeros((m, m))
        A_transition = self.get_A_s_s_joint(sa)
        for topic in self.tnb.get_topic_set(include_special_states=True):
            pair = (topic, sa)
            T_pair_s = self.tnb.get_T_y_joint(topic, sa)
            cur_lambda_joint = self.lambda_joint_s[pair]
            numerator = A_transition * T_pair_s * cur_lambda_joint * cur_lambda_joint.T * A_transition
            denom = 1 + T_pair_s * cur_lambda_joint.T * A_transition * cur_lambda_joint
            H_inv += numerator / denom
        H_inv = A_transition - H_inv
        return H_inv

    def get_H_inv_transition_y_s_joint(self, topic):
        # speech acts to topics
        m = self.tnb.num_speech_acts
        H_inv = numpy.zeros((m, m))
        A_transition = self.get_A_y_s_joint(topic)
        for sa in self.tnb.get_speech_act_set(include_special_states=True):
            pair = (topic, sa)
            T_pair_s = self.tnb.pairs_to_sa_transition_counts[pair]
            # expected transition probabilities, under current model
            cur_lambda_joint = self.lambda_joint_s[pair]
            numerator = A_transition * T_pair_s * cur_lambda_joint * cur_lambda_joint.T * A_transition
            denom = 1 + T_pair_s * cur_lambda_joint.T * A_transition * cur_lambda_joint
            H_inv += numerator / denom
        H_inv = A_transition - H_inv
        return H_inv


    def joint_sequential_log_likelihood(self, rebuild_transition_matrix=True):
        '''
        calculate the joint LL of the training data (i.e., the 
        data comprising the cases to which this model has access).
        '''
        if rebuild_transition_matrix:
            self.build_transition_probs_matrix_joint()

        # cache the normalizing constants for every
        # topic/sa pair.
        joint_Zs = self.calc_joint_Zs()

        total_log_prob = 0.0
        for X, Y, S in zip(self.tnb.cases_X, self.tnb.cases_Y, self.tnb.cases_S):
            prev_y, prev_s = Y[0], S[0] # START
            for X_j, y_j, s_j in zip(X[1:-1], Y[1:-1], S[1:-1]):
                from_pair = (prev_y, prev_s)
                from_pair_index = self.tnb.pairs_to_indices[from_pair]

                to_pair = (y_j, s_j)
                to_pair_index = self.tnb.pairs_to_indices[to_pair]

                # remember, this is a log probability
                trans_p = self.Sigma[from_pair_index, to_pair_index]
                # now calculate the emission probability
                z_pair = joint_Zs[(y_j, s_j)]
                log_p_of_x_given_y_and_s = self.p_of_inst_given_y_and_s(X_j, y_j, s_j, z=z_pair)
                total_log_prob += trans_p + log_p_of_x_given_y_and_s

                prev_y, prev_s = y_j, s_j

        return total_log_prob

    def p_of_inst_given_y_and_s(self, inst, y, s, z=None):
        '''
        calculate and return the log probability of the 
        instance x, given the topic y and speech act s.
        '''
        if z is None:
            z = self.get_joint_z(y, s)

        log_inst_prob = 0.0
        for f_j, times_observed in inst.items():
            raw_prob = numpy.exp(\
                    self.tnb.m[f_j] +\
                    self.topic_etas[y][f_j] +\
                    self.speech_act_etas[s][f_j] +\
                    self.topic_sa_interaction_etas[(y,s)][f_j])/z
            log_prob = numpy.log(raw_prob)
            log_prob = sum([log_prob]*times_observed)
            log_inst_prob += log_prob
        return log_inst_prob


    def build_transition_probs_matrix_joint(self):
        '''
        build a *joint* transition probability matrix 
        reflecting transition likelihoods from and to 
        (topic, speech act) pairs. operationally, we will
        accomplish this by setting up a relatively big
        (topics * speech acts) x (topics * speech acts) matrix.

        the indexing will be as follows. a given topic, speech act
        pair will be assigned an index:
            topic_index + (speech_act_index + 1)
        '''
        num_states = len(self.tnb.topic_sa_pairs)

        self.Sigma = \
            numpy.matrix(numpy.zeros((num_states, num_states)))

        # we never transition from the STOP state to anywhere.
        from_pairs =\
            [pair for pair in self.tnb.topic_sa_pairs if not "STOP" in pair]

        # and we never transition *into* the START state.
        to_pairs =\
            [pair for pair in self.tnb.topic_sa_pairs if not "START" in pair]

        for from_pair in from_pairs:
            # the normalization factors are the sums of the
            # transition probabilities leaving this pair to
            # topics and speech acts, respectively
            from_pair_index = self.tnb.pairs_to_indices[from_pair]
            prev_topic, prev_sa = from_pair

            # normalization terms out of this pair into
            # topics and speech acts.
            z_topic = self.z_for_joint_to_y(prev_topic, prev_sa)
            z_sa = self.z_for_joint_to_s(prev_topic, prev_sa)

            for to_pair in to_pairs:
                to_pair_index = self.tnb.pairs_to_indices[to_pair]

                y, s = to_pair
                pair_to_y_prob = self.y_transition_prob_joint(y, prev_topic, prev_sa, z_topic)
                pair_to_s_prob = self.s_transition_prob_joint(s, prev_topic, prev_sa, z_sa)
                
                # note -- Sigma holds *log* probabilities
                joint_transition_p = numpy.log(pair_to_y_prob) + numpy.log(pair_to_s_prob)
                self.Sigma[from_pair_index, to_pair_index] = joint_transition_p
                

    def z_for_joint_to_y(self, topic, speech_act):
        ''' 
        normalization term for transitions leaving 
        (topic, speech_act) and heading to other topics.
        '''
        z = 0.0
        # sum over probabilities from this pair to all
        # topics
        for dest_topic in self.tnb.get_topic_set(include_special_states=True, exclude_stop_state=True):
            dest_topic_index = self.tnb.topics_to_indices[dest_topic]
            log_transition_prob =\
                self.tnb.pi_topic[dest_topic_index] + \
                self.sigma_y_y[topic][dest_topic_index] +\
                self.sigma_s_y[speech_act][dest_topic_index] +\
                self.sigma_interactions_y[(topic, speech_act)][dest_topic_index]

            z += numpy.exp(log_transition_prob)
        return z

    def z_for_joint_to_s(self, topic, speech_act):
        ''' 
        normalization term for transitions leaving 
        (topic, speech_act) and heading to other topics.
        '''
        z = 0.0
        #for dest_topic, dest_speech_act in self.tnb.topic_sa_pairs:
        for dest_sa in self.tnb.get_speech_act_set(include_special_states=True, exclude_stop_state=True):
            dest_sa_index = self.tnb.speech_acts_to_indices[dest_sa]
            log_transition_prob =\
                self.tnb.pi_sa[dest_sa_index] + \
                self.sigma_s_s[speech_act][dest_sa_index] +\
                self.sigma_y_s[topic][dest_sa_index]+\
                self.sigma_interactions_s[(topic, speech_act)][dest_sa_index]

            z += numpy.exp(log_transition_prob)
        return z

    def build_transition_probs_matrix_y(self):
        ''' 
            this method assumes the model parameters are fit, and then 
            generates a state x state transition probability matrix for 
            convienence later on. 
        '''
        ### 
        # we will use the convention that rows represent
        # *from* and columns are the topics transitioned
        # *to*. we use ones -- rather than zeros -- as 
        # a psuedo-count ('laplace smoothing')
        num_states = self.tnb.num_topics # @TODO speech acts, too!
        self.Sigma = \
            numpy.matrix(numpy.zeros((num_states, num_states)))

        y_y_Zs = self.get_y_y_transition_Zs()

        for from_topic in self.tnb.get_topic_set(include_special_states=True, exclude_stop_state=True):
            from_topic_index = self.tnb.topics_to_indices[from_topic]
            z = y_y_Zs[from_topic]
            for to_topic in self.tnb.get_topic_set(include_special_states=True):
                to_topic_index = self.tnb.topics_to_indices[to_topic]
                y_y_transition_prob = self.y_y_transition_prob(to_topic, from_topic, z)
                #### we should maybe log-these??
                self.Sigma[from_topic_index,to_topic_index] = y_y_transition_prob        
      

    '''''''''''''''''''''''''''''''''''''''
                prediction stuff
    '''''''''''''''''''''''''''''''''''''''
    def _get_set(self, l):
        # flatten list of lists l; return a set
        # of unique elements therein
        lbl_set = []
        for l_i in l:
            lbl_set.extend(l_i)
        return set(lbl_set)

    def viterbi_sequence_predict_joint(self, X, rebuild_transition_matrix=True, joint_Zs=None):
        if rebuild_transition_matrix:
            self.build_transition_probs_matrix_joint()

        V = [{}] 
        path = {}

        ## time 0
        for topic in self.tnb.get_topic_set(include_special_states=True):
            V[0][topic] = 0.0
            path[topic] = [topic]
        
        path[("START", "START")] = [("START", "START")]
        V[0][("START", "START")] = 1.0

        # and we never transition *into* the START state.
        all_pairs =\
            [pair for pair in self.tnb.topic_sa_pairs if not "START" in pair and not "STOP" in pair]

        for i,x in enumerate(X[1:-1]): 
            V.append({})
            new_path = {}
            
            t = i+1 # +1 for the "START" offset
            ### @TODO modify for joint
            for pair in all_pairs:
                y, s = pair
                z = joint_Zs[pair]
                log_emission_p = self.p_of_inst_given_y_and_s(x, y, s, z=z)
                pair_index = self.tnb.pairs_to_indices[pair]
                
                best_state = None
                best_state_log_prob = float("-inf")
                
                # on the very first step, we transition from the "START"
                # state so we need only consider this state.
                prev_states_to_consider = all_pairs
                if t == 1:
                    prev_states_to_consider = [("START", "START")]
                
                for y0, s0 in prev_states_to_consider:
                    '''
                    (prob, state) = max([(V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states])
                    '''
                    pair0 = (y0, s0)
                    pair0_index = self.tnb.pairs_to_indices[pair0]
                    # remember, the Sigma's in the joint case are *already* logged.
                    
                    cur_log_prob = V[t-1][pair0] + self.Sigma[pair0_index, pair_index] + log_emission_p
                    
                    if cur_log_prob > best_state_log_prob:
                        best_state = pair0
                        best_state_log_prob = cur_log_prob

                V[t][pair] = best_state_log_prob
                if pair is None or best_state is None:
                    pdb.set_trace()
                new_path[pair] = path[best_state] + [pair]

            path = new_path
        
        # now transition to the end state 
        end_pair = ("STOP", "STOP")
        end_pair_index = self.tnb.pairs_to_indices[end_pair]
        V.append({})
        best_state, best_log_prob, best_path = float("-inf"), float("-inf"), None
        for pair0 in all_pairs:
            pair0_index = self.tnb.pairs_to_indices[pair0]
            # -2 is the last index because we just appened an additional
            # element (see above)
            pair0_log_prob = V[-2][pair0] + self.Sigma[pair0_index, end_pair_index]
            V[-2][pair0] = pair0_log_prob
            path[pair0] = path[pair0] + [end_pair] # append stop states
            if pair0_log_prob > best_log_prob:
                best_state = pair0
                best_path = path[pair0]
                best_log_prob = pair0_log_prob
        
        return best_path

    def predict_set_sequential_joint(self, test_set):
        '''
        for each sequence (case) in test_set, jointly predict
        its topics and speech acts.
        '''
        joint_Zs = None
        # calculate the normalization terms up-front
        joint_Zs = self.calc_joint_Zs()

        all_preds_y, all_preds_s = [], []
        for i, seq in enumerate(test_set):
            print "on case {0}".format(i)
            preds = self.viterbi_sequence_predict_joint(seq, joint_Zs=joint_Zs)
            # the joint method returns a set of sequential
            # tuples (y, s) where y is a topic and s is a 
            # speech act -- so we parse these out, respectively
            topic_preds, sa_preds = [], []
            for pred in preds:
                topic_preds.append(pred[0])
                sa_preds.append(pred[1])

            all_preds_y.append(topic_preds)
            all_preds_s.append(sa_preds)
            
        return all_preds_y, all_preds_s
