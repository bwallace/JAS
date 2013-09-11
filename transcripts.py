'''
Byron C. Wallace

This module defines objects and methods for representing and processing 
GMIAS annotated patient-doctor transcripts. It provides functionality to
parse encoded transcripts and helper methods providing count statistics.

Requires: scipy, numpy

Example usage:

    transcripts = \
        transcripts.tnb_from_file("data/unigram-cases-joint/train.CRF.speakers.pronoun.question.unigram.joint.0.dat")

'''

import pdb 
import itertools
import random

import scipy
import numpy
from numpy import arange, ones
from scipy.sparse import coo_matrix, dok_matrix, lil_matrix

# representing the beginning and end of visits.
SPECIAL_STATES =  set(["START", "STOP"])

class JointModel:

    def __init__(self, cases, hold_out_a_set=False):
        self.cases = cases
        self.all_cases = list(cases) # including any hold-out cases
        self.num_cases = len(cases)

        '''
        this deals with taking a percentage of the training
        data (cases) and holding it out for model tuning.
        note that this is a subset of *training* data, 
        and has nothing to do with the test set.
        '''
        self.hold_out_a_set = hold_out_a_set
        self.hold_out_p = .05 # 5% of the cases (if any)
        self.hold_out_indices = []
        self.held_out_cases = []
        self.held_out_cases_X, self.held_out_cases_Y, self.held_out_cases_S = [], [], []
        if self.hold_out_a_set:
            n = int(self.hold_out_p * self.num_cases)
            # these indices will refer to the hold out cases
            self.hold_out_indices = random.sample(range(self.num_cases), n)
            train_cases = []
            for i, case in enumerate(self.cases):
                if i in self.hold_out_indices:
                    self.held_out_cases.append(case)
                else:
                    train_cases.append(case)
            self.cases = train_cases
            self.num_cases = len(train_cases)
            print "-- holding out {0} cases!".format(n)

        # grab all unique codes
        self.topic_set, self.speech_act_set = [], []
        for c in self.all_cases:
            self.topic_set.extend(c.get_unique_topic_set())
            self.speech_act_set.extend(c.get_unique_speech_act_set())

        self.topic_set = set(self.topic_set)
       
        self.speech_act_set = set(self.speech_act_set)
        
        print "{0} unique topic codes; {1} unique speech act codes -- **excluding special states**".\
                        format(len(self.topic_set), len(self.speech_act_set))

        ### *including* special states
        self.num_topics = len(self.topic_set.union(SPECIAL_STATES))
        self.num_speech_acts = len(self.speech_act_set.union(SPECIAL_STATES))

        # (this sets some class variables)
        self.tokens_to_indices = None # array indexing, ack.
        self.topics_to_indices = {}
        self.topic_indices_to_topics = {} # backwards-lookup
        self.speech_acts_to_indices = {}
        self.speech_act_indices_to_speech_acts = {}

        # this enumerates all pairs of (topic, speech_act); useful to keep around
        self.topic_sa_pairs = list(itertools.product(self.topic_set.union(SPECIAL_STATES), \
                                                        self.speech_act_set.union(SPECIAL_STATES)))
        # for convienence, construct a dictionary mapping pairs
        # to indices (e.g., to be used in the joint transition matrix)
        self.pairs_to_indices = {}
        self.pair_indices_to_pairs = {} # keep a pointer the other way, too.
        self.pair_counts = {} # keep counts of all pair occurences
        self.pair_C_d = {} # cache token counts for pairs
        for i, pair in enumerate(self.topic_sa_pairs):
            self.pairs_to_indices[pair] = i
            self.pair_indices_to_pairs[i] = pair
            self.pair_counts[pair] = 0

        self.count_tokens()
        self.C_topic_d = {}
        self.C_sa_d = {}
        # this dictionary holds word counts for
        # pairs of topics/speech acts
        self.C_t_sa_d = {}

        self.topic_frequencies = numpy.zeros(len(self.topic_set.union(SPECIAL_STATES)))
        self.speech_act_frequencies = numpy.zeros(len(self.speech_act_set.union(SPECIAL_STATES)))

        self.calculate_vals_for_model()


    def get_topic_set(self, include_special_states=False, exclude_stop_state=False):
        if include_special_states:
            if not exclude_stop_state:
                return self.topic_set.union(SPECIAL_STATES)
            else:
                # sometimes we don't want to deal with the "STOP" state, which
                # can be sort of a nuisance
                return [y for y in self.topic_set.union(SPECIAL_STATES) if y != "STOP"]

        return self.topic_set

    def get_speech_act_set(self, include_special_states=False, exclude_stop_state=False):
        if include_special_states:
            if not exclude_stop_state:
                return self.speech_act_set.union(SPECIAL_STATES)
            else:
                return [y for y in self.speech_act_set.union(SPECIAL_STATES) if y != "STOP"]

        return self.speech_act_set

    '''
    the following methods provide functionality to interface
    with the joint-sequential stuff.

    this includes mapping the parsed pythonic data formats 
    (e.g., dictionaries) into vector representations for consumption 
    by the joint_sequential model, and re-scaling probability estimates, 
    etc.
    '''
    def calculate_vals_for_model(self):
        # get_vector_indices maps tokens and state names to vector 
        # indices; these are stored in instance dictionaries.
        self.get_vector_indices()

        # now populate the instances and Y,S
        # labels.
        self.instances = []
        self.Y = [] # topic codes
        self.S = [] # speech act codes
        print "populating instances..."
        self.populate_instances()
        print "ok."
        print "calculating background word (log) frequencies ..."
        self.m = self.get_log_frequencies()

        print "calculating background topic and speech act (log) frequencies"
        self.pi_topic = self.get_log_topic_frequences()
        self.pi_sa = self.get_log_sa_frequencies()
        print "counting transitions..."
        self.count_transitions()
        print "all set."    
        

    def get_vector_indices(self):
        '''
        this method maps tokens to vector indices, because
        (depending on the train/test split, etc.) the token 'ids' 
        themselves sometimes either skip counts (e.g., start 
        with 3).
        '''
        # only set this once.
        if self.tokens_to_indices is None: 
            self.tokens_to_indices = {}
            for i, w in enumerate(self.all_tokens):
                # this means i maps to w.
                self.tokens_to_indices[w] = i


            for topic_i, topic in enumerate(self.topic_set.union(SPECIAL_STATES)):
                self.topics_to_indices[topic] = topic_i
                self.topic_indices_to_topics[topic_i] = topic # keep a reverse mapping, too.

            for sa_i, sa in enumerate(self.speech_act_set.union(SPECIAL_STATES)):
                self.speech_acts_to_indices[sa] = sa_i
                self.speech_act_indices_to_speech_acts[sa_i] = sa


    def populate_instances(self, joint=False):
        self.cases_X, self.cases_Y, self.cases_S = [], [] , []
       
        for case_i, case in enumerate(self.all_cases):
            if case_i % 5 == 0:
                print "on case {0}".format(case_i)

            cur_case_X = [{}] # empty observation for start state.
            cur_case_Y, cur_case_S = [["START"], ["START"]]
            for utterance in case.utterances:
                # again, we have changed the indices (see above)
                # so here we build a dictionary reflecting these
                d_with_updated_keys = self._to_d(utterance)

                if not case_i in self.hold_out_indices:
                    self.instances.append(d_with_updated_keys)
                    self.Y.append(utterance.topic)
                    self.S.append(utterance.speech_act)

                cur_case_X.append(d_with_updated_keys)
                cur_case_Y.append(utterance.topic)
                cur_case_S.append(utterance.speech_act)

            cur_case_X.append({})
            cur_case_Y.append("STOP")
            cur_case_S.append("STOP")

            if not case_i in self.hold_out_indices:           
                self.cases_X.append(cur_case_X)
                self.cases_S.append(cur_case_S)
                self.cases_Y.append(cur_case_Y)
            else:
                print "-- holding out case {0}".format(case_i)
                self.held_out_cases_X.append(cur_case_X)
                self.held_out_cases_Y.append(cur_case_Y)
                self.held_out_cases_S.append(cur_case_S)

    def _to_d(self, utterance):
        '''
        map the tokens comprising utterance to a 
        dictionary in which the keys reflect the
        updated indexes (as per self.tokens_to_indices)
        '''
        d = {}
        for w in utterance:
            try:
                i = self.tokens_to_indices[w]
                if i in d:
                    d[i] += 1
                else:
                    d[i] = 1
            except:
                print "word {0} not in vocabularly; ignoring".format(w)
        return d

    def get_log_topic_frequences(self, norm=True):
        log_topic_freqs = numpy.zeros(self.num_topics)
        for topic in self.topic_set.union(SPECIAL_STATES):
            topic_count = None
            if topic in SPECIAL_STATES:
                topic_count = self.num_cases
            else:
                topic_count = self.Y.count(topic)
            i = self.topics_to_indices[topic]
            log_topic_freqs[i] = topic_count

        z = sum(log_topic_freqs) if norm else 1.0
        for i in xrange(len(log_topic_freqs)):
            log_topic_freqs[i] = numpy.log(log_topic_freqs[i] / z)

        return log_topic_freqs

    def get_log_sa_frequencies(self, norm=True):
        log_sa_freqs = numpy.zeros(self.num_speech_acts)
        for sa in self.speech_act_set.union(SPECIAL_STATES):
            sa_count = None
            if sa in SPECIAL_STATES:
                sa_count = self.num_cases
            else:
                sa_count = self.S.count(sa)
            # get the speech-act index
            i = self.speech_acts_to_indices[sa]
            log_sa_freqs[i] = sa_count

        z = sum(log_sa_freqs) if norm else 1.0
        for i in xrange(len(log_sa_freqs)):
            log_sa_freqs[i] = numpy.log(log_sa_freqs[i] / z)

        return log_sa_freqs

    def get_log_frequencies(self):
        ''' calculate the background distribution (over vocab) '''
        m = numpy.zeros(self.num_tokens)
        for w in self.all_tokens:
            i = self.tokens_to_indices[w]
            m[i] = numpy.log(self.background_token_counts[w])
        return m

    def get_c_pair(self, pair):
        if pair in self.pair_C_d:
            return self.pair_C_d[pair]

        pair_w_counts = self.joint_token_counts[pair]
        c_pair = numpy.zeros(self.num_tokens)
        for w, w_count in pair_w_counts.items():
            i = self.tokens_to_indices[w]
            c_pair[i] = w_count
        self.pair_C_d[pair] = c_pair
        return c_pair

    def get_C_topic(self, topic):
        ''' returns the *sum* of the count vectors for <topic> '''
        if topic in self.C_topic_d:
            return sum(self.C_topic_d[topic])

        self.get_c_topic(topic) # calculate and cache counts
        return sum(self.C_topic_d[topic])

    def get_c_topic(self, topic):
        '''
        C_topic is the vector of summed term counts
        for <topic>.
        '''
        if topic in self.C_topic_d:
            return self.C_topic_d[topic]

        topic_token_counts = self.topic_token_counts[topic]
        C_topic = numpy.zeros(self.num_tokens)
        for w, w_count in topic_token_counts.items():
            i = self.tokens_to_indices[w] # get the vector index for this topic
            C_topic[i] = w_count

        self.C_topic_d[topic] = C_topic
        return C_topic

    def get_C_speech_act(self, speech_act):
        '''
        C is the sum of the count vectors; this is 
        just a convenience method.
        '''
        if speech_act in self.C_sa_d:
            return sum(self.C_sa_d[speech_act])
        self.get_c_speech_act(speech_act)
        return sum(self.C_sa_d[speech_act])

    def get_c_speech_act(self, speech_act):
        '''
        c_speech_act is the vector of summed term counts
        for <speech_act>.
        '''
        if speech_act in self.C_sa_d:
            return self.C_sa_d[speech_act]

        sa_token_counts = self.speech_act_token_counts[speech_act]
        C_sa = numpy.zeros(self.num_tokens)
        for w, w_count in sa_token_counts.items():
            i = self.tokens_to_indices[w] # get the vector index for this topic
            C_sa[i] = w_count
        self.C_sa_d[speech_act] = C_sa
        return C_sa

    def get_C_joint(self, topic, speech_act):
        if (topic, speech_act) in self.C_t_sa_d:
            return sum(self.C_t_sa_d[(topic, speech_act)])
        self.get_c_joint(topic, speech_act)
        return sum(self.C_t_sa_d[(topic, speech_act)])

    def get_c_joint(self, topic, speech_act):
        '''
        get a vector of word counts for cases where
        topic=topic *and* speech_act=speech_act.
        '''
        if (topic, speech_act) in self.C_t_sa_d:
            return self.C_t_sa_d[(topic, speech_act)]

        t_sa_token_counts = self.joint_token_counts[(topic, speech_act)]
        C_t_sa = numpy.zeros(self.num_tokens)
        for w, w_count in t_sa_token_counts.items():
            i = self.tokens_to_indices[w]
            C_t_sa[i] = w_count
        self.C_t_sa_d[(topic, speech_act)] = C_t_sa
        return C_t_sa


    def get_dimensions(self):
        return self.num_tokens

    '''
    some helper methods for grabbing the transition counts.
    recall that y = topics; s = speech acts; little t will
    return count vectors; big T sums
    '''
    ###
    # from topics to topics
    def get_T_y_y(self, topic):
        return sum(self.topic_to_topic_transition_counts[topic])

    def get_t_y_y(self, topic):
        return self.topic_to_topic_transition_counts[topic]

    ###
    # from topics to speech acts
    def get_T_y_s(self, topic):
        return sum(self.topic_to_sa_transition_counts[topic])

    def get_t_y_s(self, topic):
        return self.topic_to_sa_transition_counts[topic]

    ###
    # from speech acts to speech acts
    def get_T_s_s(self, sa):
        return sum(self.sa_to_sa_transition_counts[sa])

    def get_t_s_s(self, sa):
        return self.sa_to_sa_transition_counts[sa]

    ###
    # from speech acts to topics
    def get_T_s_y(self, sa):
        return sum(self.sa_to_topic_transition_counts[sa])

    def get_t_s_y(self, sa):
        return self.sa_to_topic_transition_counts[sa]

    ### and for the joint stuff!
    def get_T_y_joint(self, from_topic, from_speech_act):
        '''
        get a vector of transition counts to *topics* where
        the origin state had topic from_topic and speech act
        speech_act.
        '''
        return sum(self.pairs_to_topic_transition_counts[(from_topic, from_speech_act)])

    def get_t_y_joint(self, from_topic, from_speech_act):
        return self.pairs_to_topic_transition_counts[(from_topic, from_speech_act)]

    def get_T_s_joint(self, from_topic, from_speech_act):
        '''
        get a vector of transition counts to *topics* where
        the origin state had topic from_topic and speech act
        speech_act. 

        note that this will be the same number as get_T_y_joint
        (since we transition out of this pair the same number of times)
        '''
        return sum(self.pairs_to_sa_transition_counts[(from_topic, from_speech_act)])

    def get_t_s_joint(self, from_topic, from_speech_act):
        return self.pairs_to_sa_transition_counts[(from_topic, from_speech_act)]


    #### end helper methods #####

    def count_transitions(self):
        '''
        counts up the transitions for each topic and speech act;
        these are held in class variables. we also count up the
        (topic, speech act) state frequencies, in general.

        we also keep *joint* counts reflecting transitions from pairs of 
        states to topics and speech_acts

        note that we manually insert the special "START", "STOP" 
        states here.
        '''
        # obviously could be generalized (and made more terse) with matrices, 
        # but I find keeping explicit dictionaries around most readable.
        self.topic_to_topic_transition_counts, self.sa_to_topic_transition_counts = {}, {}
        self.sa_to_sa_transition_counts, self.topic_to_sa_transition_counts = {}, {}

        self.pairs_to_topic_transition_counts, self.pairs_to_sa_transition_counts = {}, {}


        # initialize transition counts out of the topic act states.
        for topic in self.topic_set.union(SPECIAL_STATES):
            self.topic_to_topic_transition_counts[topic] = numpy.zeros(len(self.topic_set.union(SPECIAL_STATES)))
            self.topic_to_sa_transition_counts[topic] = numpy.zeros(len(self.speech_act_set.union(SPECIAL_STATES)))            
            # add a transition from each pair to this topic
            for pair in self.topic_sa_pairs:
                self.pairs_to_topic_transition_counts[pair] = numpy.zeros(len(self.topic_set.union(SPECIAL_STATES)))

        # and for transitions out of speech act states.
        for sa in self.speech_act_set.union(SPECIAL_STATES):
            self.sa_to_sa_transition_counts[sa] = numpy.zeros(len(self.speech_act_set.union(SPECIAL_STATES)))
            self.sa_to_topic_transition_counts[sa] = numpy.zeros(len(self.topic_set.union(SPECIAL_STATES))) 
            # add a transition from each pair to this speech act
            for pair in self.topic_sa_pairs:
                self.pairs_to_sa_transition_counts[pair] = numpy.zeros(len(self.speech_act_set.union(SPECIAL_STATES)))

        # now count them up.
        for case in self.cases:
            # 'previous' state for each case is the special
            # START state.
            prev_topic, prev_sa = "START", "START"
            self.topic_frequencies[self.topics_to_indices["START"]] += 1
            self.speech_act_frequencies[self.speech_acts_to_indices["START"]] += 1

            for utterance in case.utterances:
                cur_topic, cur_sa = utterance.topic, utterance.speech_act
                cur_topic_index = self.topics_to_indices[cur_topic]
                cur_sa_index = self.speech_acts_to_indices[cur_sa]

                # update counts reflecting transitions into the current topic
                self.topic_to_topic_transition_counts[prev_topic][cur_topic_index] += 1
                self.sa_to_topic_transition_counts[prev_sa][cur_topic_index] += 1 

                # and now for the current speech act.
                self.sa_to_sa_transition_counts[prev_sa][cur_sa_index] += 1
                self.topic_to_sa_transition_counts[prev_topic][cur_sa_index] += 1

                # update pair counts
                cur_pair = (prev_topic, prev_sa) 
                self.pairs_to_topic_transition_counts[cur_pair][cur_topic_index] += 1
                self.pairs_to_sa_transition_counts[cur_pair][cur_sa_index] += 1

                prev_topic, prev_sa = cur_topic, cur_sa

            # finally, count transitions into the 'end state'
            topic_end_i, sa_end_i = self.topics_to_indices["STOP"], self.speech_acts_to_indices["STOP"]
            self.topic_to_topic_transition_counts[prev_topic][topic_end_i] += 1
            self.sa_to_topic_transition_counts[prev_sa][topic_end_i] += 1 
            self.sa_to_sa_transition_counts[prev_sa][sa_end_i] += 1
            self.topic_to_sa_transition_counts[prev_topic][sa_end_i] += 1

            cur_pair = (prev_topic, prev_sa) 
            self.pairs_to_topic_transition_counts[cur_pair][topic_end_i] += 1
            self.pairs_to_sa_transition_counts[cur_pair][sa_end_i] += 1

            self.topic_frequencies[self.topics_to_indices["STOP"]] += 1
            self.speech_act_frequencies[self.speech_acts_to_indices["STOP"]] += 1

    def count_tokens(self):
        ####
        # gather token set.
        self.all_tokens = set()
        for case in self.cases:
            self.all_tokens = self.all_tokens.union(case.get_all_tokens())
        self.num_tokens = len(self.all_tokens)

        print "total tokens: {0}".format(self.num_tokens)
        # just to make sure ordering is preserved...
        self.all_tokens = list(self.all_tokens)


        # this will hold the token counts *regardless
        # of labels*, i.e., the 'background' distribution
        # of tokens
        self.background_token_counts = {} 

        # this will hold the token counts conditioned
        # on speech acts
        self.speech_act_token_counts = {}
        for sa in self.speech_act_set:
            self.speech_act_token_counts[sa] = {}

        # and this will hold the token counts conditioned
        # on topic
        self.topic_token_counts = {}
        for topic in self.topic_set:
            self.topic_token_counts[topic] = {}

        # finally, this holds the token counts 
        # jointly conditioned on topic and speech
        # act
        self.joint_token_counts = {}

        for pair in self.topic_sa_pairs :
            self.joint_token_counts[pair] = {}

        # initialize
        for token in self.all_tokens:
            self.background_token_counts[token] = 0
            for topic in self.topic_set:
                self.topic_token_counts[topic][token] = 0
            for sa in self.speech_act_set:
                self.speech_act_token_counts[sa][token] = 0
            # joint counts
            for pair in self.topic_sa_pairs:
                self.joint_token_counts[pair][token] = 0

        # now count.
        for case in self.cases:
            case_token_counts = case.get_token_counts()
            for t in case_token_counts:
                self.background_token_counts[t] += case_token_counts[t]

            case_topics_to_token_counts, case_speech_acts_to_token_counts, pairs_to_token_counts, pair_counts \
                                    = case.get_conditional_word_counts(self.topic_set, self.speech_act_set)

            # add pair counts 
            for pair, count in pair_counts.items():
                self.pair_counts[pair] += count

            # add counts from this case to the sum for the topic-conditional counts
            for case_topic, topic_token_counts in case_topics_to_token_counts.items():
                for case_token in topic_token_counts:
                    self.topic_token_counts[case_topic][case_token] += \
                                            topic_token_counts[case_token]
        
            # and now for the speech act-conditional counts
            for case_sa, sa_token_counts in case_speech_acts_to_token_counts.items():
                for case_token in sa_token_counts:
                    self.speech_act_token_counts[case_sa][case_token] += \
                                            sa_token_counts[case_token]

            # and for the joint
            for pair, pair_token_counts in pairs_to_token_counts.items():
                for case_token in pair_token_counts:
                    self.joint_token_counts[pair][case_token] += pair_token_counts[case_token]


        print "ok -- all counted up."

class Case:

    def __init__(self, identifier, utterances):
        self.utterances = utterances
        self.identifier = identifier
        self.N = len(self.utterances)

    def __len__(self):
        return self.N

    def __get__(self, i):
        return self.utterances[i]

    def contains_token(self, t):
        return t in self.get_all_tokens()

    def get_all_tokens(self):
        self.token_set = set()
        for u in self.utterances:
            self.token_set = self.token_set.union(u.get_token_set())
        return self.token_set

    def get_token_counts(self):
        # token counts, independent of topic/speech act
        token_counts = {}
        all_tokens = self.get_all_tokens()
        for t in all_tokens:
            token_counts[t] = 0

        for u in self.utterances:
            for t in u.tokens:
                token_counts[t] += 1
        return token_counts    

    def get_conditional_word_counts(self, topics, speech_acts):
        # we take in a parametric alphabet of topics and SA's
        # in case specific ones are unobserved
        # in this session
        topics_to_token_counts = {}
        speech_acts_to_token_counts = {}
        pairs_to_token_counts = {} # (topic, speech_act) -> counts
        pair_counts = {}

        ###
        # initialize topic-specific token count dictionary
        for topic in topics:
            topics_to_token_counts[topic] = {}
            for token in self.token_set:
                # we don't do any smoothing here, as it is
                # assumed to be done at a higher level
                topics_to_token_counts[topic][token] = 0

        # ... and the speech-act specific one
        for speech_act in speech_acts:
            speech_acts_to_token_counts[speech_act] = {}
            for token in self.token_set:
                speech_acts_to_token_counts[speech_act][token] = 0

        # and, finally, the counts specific to pairs of observations
        for pair in itertools.product(topics, speech_acts):
            pairs_to_token_counts[pair] = {}
            for token in self.token_set:
                pairs_to_token_counts[pair][token] = 0


        # count 'em up.
        for u in self.utterances:
            topic = u.topic
            sa = u.speech_act
            if (topic, sa) in pair_counts:
                pair_counts[(topic, sa)] += 1
            else:
                pair_counts[(topic, sa)] = 1

            for token in u.tokens:
                topics_to_token_counts[topic][token] += 1
                speech_acts_to_token_counts[sa][token] += 1
                pairs_to_token_counts[(topic, sa)][token] += 1

        return topics_to_token_counts, speech_acts_to_token_counts, pairs_to_token_counts, pair_counts



    def get_binned_indices(self, T):
        '''
        slices the sample in T bins, returns indices
        of this binning as follows:

            [(l0, u0), (l1, u1), ... , (lT, uT)]

        where the tuples are lower and upper indices, 
        respectively.
        '''
        slice_size = int(self.N*(1./T))
        t_index = 0
        indices = []
        for lower_index in range(0, self.N, slice_size):
            if t_index < T:
                if t_index == T-1:
                    indices.append((lower_index, self.N-1))
                else:
                    upper_index = lower_index+slice_size
                    indices.append((lower_index, upper_index))
                t_index += 1
        return indices

    def get_labels(self):
        return [u.topic for u in self.utterances]

    def get_label_counts(self, normalize=False):
        ''' 
        return counts for both topic codes and speech 
        acts.
        '''

        self.topic_ps = {}
        topic_set = self.get_unique_topic_set()
        for topic in topic_set:
            self.topic_ps[topic] = 0.0

        self.speech_act_ps = {}
        speech_act_set = self.get_unique_speech_act_set()
        for sa in speech_act_set:
            self.speech_act_ps[sa] = 0.0

        for u in self.utterances:
            self.topic_ps[u.topic] += 1.0
            self.speech_act_ps[u.speech_act] += 1.0

        # normalize
        if normalize:
            for topic in topic_set:
                self.topic_ps[topic] = self.topic_ps[topic]/float(self.N)
            for sa in speech_act_set:
                self.speech_act_ps[sa] = self.speech_act_ps[sa]/float(self.N)

        return self.topic_ps, self.speech_act_ps

    def get_unique_topic_set(self):
        return self.get_unique_label_set()

    def get_unique_speech_act_set(self):
        return self.get_unique_label_set(speech_act=True)

    def get_unique_label_set(self, speech_act=False):
        '''
        returns a list of labels observed 
        in this case
        '''
        if speech_act:
            return list(set([u.speech_act for u in self.utterances]))
        return list(set([u.topic for u in self.utterances]))

    def build_transition_matrix(self, symbols_to_indices):
        ###
        # symbols_to_indices points codes to array indices
        # to be used.
        alphabet_size = len(symbols_to_indices)
        self.A_c = \
            numpy.matrix(numpy.zeros((alphabet_size, alphabet_size)))

        # we do *not* add psuedo-counts here; this is assumed
        # to be done at some higher-level
        from_topic = "START" # the start state
        for u in self.utterances:
            # remember, we are using the convention that
            # rows are *from*, columns are *to*
            from_topic_index = symbols_to_indices[from_topic]
            to_topic_index = symbols_to_indices[u.topic]
            self.A_c[from_topic_index,to_topic_index] += 1
            # set the from topic
            from_topic = u.topic
        # now transition to the special (end) state
        from_topic_index = symbols_to_indices[from_topic]
        to_topic_index = symbols_to_indices["STOP"]
        self.A_c[from_topic_index, to_topic_index] += 1

        return self.A_c        


class Utterance:
    def __init__(self, tokens, topic, speech_act):
        self.tokens = tokens
        self.topic = topic
        self.speech_act = speech_act

    def __str__(self):
        return "tokens: {0}\ntopic: {1}\nspeech act: {2}\n".\
            format(" ".join([str(t) for t in self.tokens]), 
                    self.topic, self.speech_act)

    def __getitem__(self, i):
        return self.tokens[i] 

    def __len__(self):
        return len(self.tokens)

    def get_token_set(self):
        return set(self.tokens)


def tnb_from_file(file_path, hold_out_a_set=False):
    '''
    Parses the cases contained in the file at file_path.

    Assumes observations have the following format:

        T_0:C_0 ... T_M:C_M L1,L2 

    where T_0:C_0 denotes C_0 observations of token
    T_0. for the corresponding utterance, L1 is
    its label (topic) and L2 is its speech act code 
    (use only one for the univariate case; drop the comma)
    ''' 
    cases = cases_from_file(file_path)
    return JointModel(cases, hold_out_a_set=hold_out_a_set)

def load_test_cases(file_path, train_tnb):
    '''
    the reason we take a tnb_joint object here
    is because we use its vocabularly to encode
    the test utterances
    '''
    test_cases = cases_from_file(file_path, labels_too=False)

    test_cases_X = []
    for case_i, case in enumerate(test_cases):
        cur_case_X = [{}] # start state (empty observation)
        for utterance in case.utterances:
            # use the indices corresponding to the training
            # data set
            encoded_d = train_tnb._to_d(utterance)
            cur_case_X.append(encoded_d)
        cur_case_X.append({}) # end state
        test_cases_X.append(cur_case_X)
    return test_cases_X
           
        


def cases_from_file(file_path, labels_too=True):
    utterances = open(file_path).readlines()
    cases = get_cases(utterances, labels_too=labels_too)

    # filter empty/zero-length cases
    cases = [case for case in cases if len(case) > 0]
    return cases


def parse_labels_file(labels_file_path):
    '''
    grab the (joint) labels from the file.
    assumes these look like:

        boundary
        topic,speech_act
        ...
        boundary
        topic,speech_act
        ...
    '''
    cases_Ys = [] # list of lists; one list per case
    cases_Ss = []

    cur_case_Y, cur_case_S = ["START"], ["START"]
    for i, line in enumerate(open(labels_file_path).readlines()):
        if "boundary" in line:
            if i > 0:
                cur_case_Y.append("STOP")
                cur_case_S.append("STOP")
                ###
                # for now just doing Y; will want to use line 
                # below, instead, to parse speech act codes, too
                #cases_lbls.append(zip(cur_case_Y, cur_case_S))
                #cases_lbls.append(cur_case_Y)
                cases_Ys.append(cur_case_Y)
                cases_Ss.append(cur_case_S)
                # reset
                cur_case_Y, cur_case_S = ["START"], ["START"]
        elif not _is_blank_line(line):
            y, s = line.strip().split(",")

            # cast to ints, when possible
            y = to_int(y)
            s = to_int(s)

            cur_case_Y.append(y)
            cur_case_S.append(s)
    return cases_Ys, cases_Ss


def to_int(s):
    if not s in ["START", "STOP"]:
        return int(s)
    return s
def get_cases(utterances, labels_too=True):
    cases, cur_case = [], []
    for i,u in enumerate(utterances):
        lbl1, lbl2 = '', ''
        if "boundary" in u:
            lbl1 = lbl2 = "boundary"
        elif labels_too:
            tokens, labels = parse_utterance(u)
            lbl1, lbl2 = labels # topic, speech_act
        else:
            # only parses out utterance -- ASSUMES THERE 
            # IS NOT LABEL IN THE FILE. if there is, then
            # it will be *wrongly* included in the tokens,
            # below.
            #pdb.set_trace()
            tokens = parse_utterance(u, labels_too=False)
        
        if "boundary" in lbl1:
            # switch to a new case
            if not i == 0: # first case
                cases.append(Case(i, cur_case))
            
            cur_case = []
        else:
            discretized_tokens = [int(t_i) for t_i in tokens]
            if labels_too:
                # note that we're parsing both the topic and speech act label
                cur_case.append(Utterance(discretized_tokens, int(lbl1), int(lbl2)))
            else:
                cur_case.append(Utterance(discretized_tokens,  None, None))
                #pdb.set_trace()
    if len(cur_case) > 0:
        # last case!
        cases.append(Case(i, cur_case))

    return cases

def parse_utterance(u, labels_too=True, skip_intercept=True):
    tokenized = u.split(" ")
    tokens = [u_i.strip() for u_i in tokenized if not _is_blank_line(u_i)]
    if skip_intercept:
        tokens = tokens[1:]

    if labels_too:
        # parse *both* labels!
        # in our case, this is topic, speech_act
        lbl1, lbl2 = tokens[-1].strip().split(",") 
        tokens = tokens[:-1] # trim the label, which comes last
        return (tokens, (lbl1, lbl2))
    # if we're not grabbing labels (e.g., because we are pulling
    # cases from a 'test' file, which are sans labels) then
    # then just return tokens
    return tokens

def _is_blank_line(line):
    return line.strip() in ("")