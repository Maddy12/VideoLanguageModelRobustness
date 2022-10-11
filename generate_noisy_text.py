from styleformer import Styleformer
import logging
import argparse
import pdb
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
import os
import random
import pickle
from tqdm import tqdm
from transformers import pipeline
# import openai
import nltk
try:
    from textflint.input.component.sample.ut_sample import UTSample
    from textflint.input.component.sample.pos_sample import POSSample
    from textflint.input.component.sample.sa_sample import SASample
    from textflint.input.component.sample.mrc_sample import MRCSample
    import textflint.generation.transformation.UT as universal
    import textflint.generation.transformation.POS as pos
    import textflint.generation.transformation.SA as sentiment
    import textflint.generation.transformation.MRC as machine
except ModuleNotFoundError:
    print("Text packages not installed, if running text related functions, please install. ")
    pass

warnings.filterwarnings("ignore")


def get_gender_mappings():
    """
    This is my lazy hack. It is very specific to MSRVTT so with other datasets it might miss loads of gender specific
        wordings.
    :return:
    """
    return {'he': {'type': 'male',
                   'flip': 'she',
                   'remove': 'they'},
            'she': {'type': 'female',
                    'flip': 'he',
                    'remove': 'they'},
            'his': {'type': 'male',
                    'flip': 'hers',
                    'remove': 'theirs'},
            'hers': {'type': 'female',
                     'flip': 'his',
                     'remove': 'theirs'},
            'man': {'type': 'male',
                    'flip': 'woman',
                    'remove': 'person'},
            'woman': {'type': 'female',
                      'flip': 'man',
                      'remove': 'person'},
            'boy': {'type': 'male',
                    'flip': 'girl',
                    'remove': 'child'},
            'girl': {'type': 'female',
                     'flip': 'boy',
                     'remove': 'child'},
            'boys': {'type': 'male',
                     'flip': 'girls',
                     'remove': 'kids'},
            'girls': {'type': 'female',
                      'flip': 'boys',
                      'remove': 'kids'},
            'guy': {'type': 'male',
                    'flip': 'lady',
                    'remove': 'person'},
            'lady': {'type': 'female',
                     'flip': 'guy',
                     'remove': 'person'},
            'men': {'type': 'male',
                    'flip': 'women',
                    'remove': 'people'},
            'women': {'type': 'female',
                      'flip': 'men',
                      'remove': 'people'},
            'guys': {'type': 'male',
                     'flip': 'ladies',
                     'remove': 'people'},
            'ladies': {'type': 'female',
                       'flip': 'guys',
                       'remove': 'people'},
            'him': {'type': 'male',
                    'flip': 'her',
                    'remove': 'they'},
            'her': {'type': 'female',
                    'flip': 'him',
                    'remove': 'they'},
            'actor': {'type': 'male',
                      'flip': 'actress',
                      'remove': 'actor'},
            'actress': {'type': 'female',
                        'flip': 'actor',
                        'remove': 'actor'},
            'female': {'type': 'male',
                      'flip': 'male',
                      'remove': 'person'},
            'male': {'type': 'female',
                        'flip': 'female',
                        'remove': 'person'}
            }


def modify_pos(df, save_dir, dataset):
    """
    Will modify based on the pos tag. Different variations:
     * Remove nouns and verbs. Testing based on context
     * Remove nouns only
     * Remove verbs only
     * Keep only nouns and verbs
     * Keep only nouns
     * Keep only verbs

    :param dataset:
    :return:
    """
    collect_dict = dict()
    # Remove pos
    collect_dict['no_nouns_and_no_verbs'] = list()
    collect_dict['no_nouns'] = list()
    collect_dict['no_verbs'] = list()
    collect_dict['random_noun'] = list()
    collect_dict['random_verb'] = list()

    # Keep pos
    collect_dict['only_nouns_and_only_verbs'] = list()
    collect_dict['only_nouns'] = list()
    collect_dict['only_verbs'] = list()

    pbar = tqdm(df.iterrows(), desc=f'{dataset}', total=len(df))
    for idx, row in pbar:
        text = row['original_text']
        no_nouns_and_no_verbs = list()
        no_nouns = list()
        no_verbs = list()
        only_nouns_and_only_verbs = list()
        only_nouns = list()
        only_verbs = list()
        random_noun = list()
        random_verb = list()

        sentence = nltk.sent_tokenize(text)
        for sent in sentence:
            tags = nltk.pos_tag(nltk.word_tokenize(sent))

        for idx, (word, tag) in enumerate(tags):
            if tag.startswith('NN'):
                only_nouns_and_only_verbs.append(word)
                only_nouns.append(word)
                no_verbs.append(word)

                random_noun.append(word)

                no_nouns.append('[UNK]')
                only_verbs.append('[UNK]')
                no_nouns_and_no_verbs.append('[UNK]')
            elif tag.startswith('VB'):
                only_nouns_and_only_verbs.append(word)
                only_verbs.append(word)
                no_nouns.append(word)

                random_verb.append(word)

                no_verbs.append('[UNK]')
                only_nouns.append('[UNK]')
                no_nouns_and_no_verbs.append('[UNK]')
            else:
                no_nouns.append(word)
                no_verbs.append(word)
                no_nouns_and_no_verbs.append(word)

                only_nouns_and_only_verbs.append('[UNK]')
                only_nouns.append('[UNK]')
                only_verbs.append('[UNK]')
        # Remove pos
        if len(random_noun) > 0:
            drop_nn = random.sample(random_noun, k=1)[0]
            drop_one_noun = [x if x != drop_nn else '[UNK]' for x in text.split() ]
        else:
            drop_one_noun = text.split()

        if len(random_verb) > 0:
            drop_vb = random.sample(random_verb, k=1)[0]
            drop_one_verb = [x if x != drop_vb else '[UNK]' for x in text.split() ]
        else:
            drop_one_verb = ['[UNK]']+text.split()[1:]

        collect_dict['random_noun'].append(' '.join(drop_one_noun))
        collect_dict['random_verb'].append(' '.join(drop_one_verb))

        collect_dict['no_nouns_and_no_verbs'].append(' '.join(no_nouns_and_no_verbs))
        collect_dict['no_nouns'].append(' '.join(no_nouns))
        collect_dict['no_verbs'].append(' '.join(no_verbs))

        # Keep pos
        collect_dict['only_nouns_and_only_verbs'].append(' '.join(only_nouns_and_only_verbs))
        collect_dict['only_nouns'].append(' '.join(only_nouns))
        collect_dict['only_verbs'].append(' '.join(only_verbs))

    for perturbation, new_texts in collect_dict.items():
        df['text'] = new_texts
        df.to_csv(os.path.join(save_dir, f'{dataset}_{perturbation}.csv'), index=False)


def modify_pos_pickle(original, save_dir):
    """
    Will modify based on the pos tag. Different variations:
     * Remove nouns and verbs. Testing based on context
     * Remove nouns only
     * Remove verbs only
     * Keep only nouns and verbs
     * Keep only nouns
     * Keep only verbs

    :param dataset:
    :return:
    """
    dataset = 'youcook2'

    collect_dict = dict()
    # Remove pos
    collect_dict['no_nouns_and_no_verbs'] = dict()
    collect_dict['no_nouns'] = dict()
    collect_dict['no_verbs'] = dict()
    collect_dict['random_noun'] = dict()
    collect_dict['random_verb'] = dict()

    # Keep pos
    collect_dict['only_nouns_and_only_verbs'] = dict()
    collect_dict['only_nouns'] = dict()
    collect_dict['only_verbs'] = dict()

    pbar = tqdm(original.items(), total=len(original))
    for video_id, annot in pbar:
        collect_dict['no_nouns_and_no_verbs'][video_id] = {k:v for k,v in annot.items()}
        collect_dict['no_nouns'][video_id] = {k:v for k,v in annot.items()}
        collect_dict['no_verbs'][video_id] = {k:v for k,v in annot.items()}

        collect_dict['random_noun'][video_id] = {k:v for k,v in annot.items()}
        collect_dict['random_verb'][video_id] = {k:v for k,v in annot.items()}

        # Keep pos
        collect_dict['only_nouns_and_only_verbs'][video_id] = {k:v for k,v in annot.items()}
        collect_dict['only_nouns'][video_id] = {k:v for k,v in annot.items()}
        collect_dict['only_verbs'][video_id] = {k:v for k,v in annot.items()}

        collect_no_nouns_and_no_verbs = list()
        collect_no_nouns = list()
        collect_no_verbs = list()
        collect_only_nouns_and_only_verbs = list()
        collect_only_nouns = list()
        collect_only_verbs = list()
        collect_random_noun = list()
        collect_random_verb = list()

        for text in annot['text']:
            sentence = nltk.sent_tokenize(text)
            for sent in sentence:
                tags = nltk.pos_tag(nltk.word_tokenize(sent))

            no_nouns_and_no_verbs = list()
            no_nouns = list()
            no_verbs = list()
            only_nouns_and_only_verbs = list()
            only_nouns = list()
            only_verbs = list()
            random_noun = list()
            random_verb = list()

            for idx, (word, tag) in enumerate(tags):
                if tag.startswith('NN'):
                    only_nouns_and_only_verbs.append(word)
                    only_nouns.append(word)
                    no_verbs.append(word)

                    random_noun.append(word)

                    no_nouns.append('[UNK]')
                    only_verbs.append('[UNK]')
                    no_nouns_and_no_verbs.append('[UNK]')
                elif tag.startswith('VB'):
                    only_nouns_and_only_verbs.append(word)
                    only_verbs.append(word)
                    no_nouns.append(word)

                    random_verb.append(word)

                    no_verbs.append('[UNK]')
                    only_nouns.append('[UNK]')
                    no_nouns_and_no_verbs.append('[UNK]')
                else:
                    no_nouns.append(word)
                    no_verbs.append(word)
                    no_nouns_and_no_verbs.append(word)

                    only_nouns_and_only_verbs.append('[UNK]')
                    only_nouns.append('[UNK]')
                    only_verbs.append('[UNK]')

            # Remove a random noun and verb
            if len(random_noun) > 0:
                drop_nn = random.sample(random_noun, k=1)[0]
                drop_one_noun = [x if x != drop_nn else '[UNK]' for x in text.split()]
            else:
                drop_one_noun = text.split()

            if len(random_verb) > 0:
                drop_vb = random.sample(random_verb, k=1)[0]
                drop_one_verb = [x if x != drop_vb else '[UNK]' for x in text.split()]
            else:
                drop_one_verb = ['[UNK]']+text.split()[1:]

            collect_random_noun.append(' '.join(drop_one_noun))
            collect_random_verb.append(' '.join(drop_one_verb))

            # Remove pos
            collect_no_nouns_and_no_verbs.append(' '.join(no_nouns_and_no_verbs))
            collect_no_nouns.append(' '.join(no_nouns))
            collect_no_verbs.append(' '.join(no_verbs))

            # Keep pos
            collect_only_nouns_and_only_verbs.append(' '.join(only_nouns_and_only_verbs))
            collect_only_nouns.append(' '.join(only_nouns))
            collect_only_verbs.append(' '.join(only_verbs))

        collect_dict['random_noun'][video_id]['text'] = collect_random_noun
        collect_dict['random_verb'][video_id]['text'] = collect_random_verb

        collect_dict['no_nouns_and_no_verbs'][video_id]['text'] = collect_no_nouns_and_no_verbs
        collect_dict['no_nouns'][video_id]['text'] = collect_no_nouns
        collect_dict['no_verbs'][video_id]['text'] = collect_no_verbs

        # Keep pos
        collect_dict['only_nouns_and_only_verbs'][video_id]['text'] = collect_only_nouns_and_only_verbs
        collect_dict['only_nouns'][video_id]['text'] = collect_only_nouns
        collect_dict['only_verbs'][video_id]['text'] = collect_only_verbs

    for perturbation, new_annot in collect_dict.items():
        with open(os.path.join(save_dir, f'youcookii_{perturbation}.pickle'), 'wb') as f:
            pickle.dump(new_annot, f)


def generate_textflint(df, save_dir, dataset, type):
    """
    Can get an overview of transformations at:
        https://github.com/textflint/textflint/blob/master/docs/user/components/transformation.md
    :param dataset:
    :param type: The transformation type: universal, pos, machine, sentiment
    :return:
    """
    modules ={'universal': universal, 'pos': pos, 'machine': machine, 'sentiment': sentiment}
    samples = {'universal': UTSample, 'pos': POSSample, 'machine': MRCSample, 'sentiment': SASample}

    assert type in samples.keys(), "Passed invalid TextFlint augmentation"
    if type == 'machine':
        raise NotImplementedError

    packages = [x for x in dir(modules[type]) if not x.startswith('__') and not x.islower()]
    package_names  = [x for x in dir(modules[type]) if not x.startswith('__') and x.islower()]

    if type == 'pos':
        treebanks = ['NN', 'VB', 'JJ', 'RB']
        packages = [packages[0], packages[0], packages[0], packages[0], packages[1]]
        package_names = [package_names[0], package_names[0], package_names[0], package_names[0], package_names[1]]
    pbar1 = tqdm(enumerate(zip(packages, package_names)), total=len(packages), position=0, leave=True)
    for idx, (func, perturbation) in pbar1:
        pbar = tqdm(df.iterrows(), desc=f'{dataset} {perturbation}', total=len(df))
        if type == 'pos' and idx < len(treebanks):
            func = getattr(modules[type], func)(treebanks[idx])
            perturbation = perturbation + '_' + treebanks[idx]
        else:
            func = getattr(modules[type], func)()
        collect_text = list()

        if os.path.exists(os.path.join(save_dir, f'{dataset}_{perturbation}.csv')):
            continue
        for idx, row in pbar:
            text = row['original_text']
            if type == 'pos':
                sentence = nltk.sent_tokenize(text)
                x = list()
                y = list()
                for sent in sentence:
                    tags = nltk.pos_tag(nltk.word_tokenize(sent))
                x = [x[0] for x in tags]
                y = [y[1] for y in tags]
                data = {'x': x, 'y': y}
            elif type == 'sentiment':
                data = {'x': text, 'y': 'pos'}
            else:
                data = {'x': text}

            try:
                sample = samples[type](data)
            except:
                pdb.set_trace()

            new_text = run_transform_v2(func, sample)

            if isinstance(new_text, list):
                new_text = ' '.join(new_text)
            collect_text.append(new_text)

        changed = np.sum(collect_text != df['original_text'].values)
        print(f"{changed}/{len(df['original_text'])} changed")
        if changed > .3 * len(df['original_text']):
            print("Saving")
            df['text'] = collect_text
            df.to_csv(os.path.join(save_dir, f'{dataset}_{perturbation}.csv'), index=False)


def generate_textflint_pickle(original, save_dir, type):
    """
    This is for the UniVl structure. It is a bit lazy because doing training data.
    Can get an overview of transformations at:
        https://github.com/textflint/textflint/blob/master/docs/user/components/transformation.md
    :param type: The transformation type: universal, pos, machine, sentiment
    :return:
    """
    modules ={'universal': universal, 'pos': pos, 'machine': machine, 'sentiment': sentiment}
    samples = {'universal': UTSample, 'pos': POSSample, 'machine': MRCSample, 'sentiment': SASample}

    assert type in samples.keys(), "Passed invalid TextFlint augmentation"
    if type == 'relation' or type == 'machine' or type == 'semantic':
        raise NotImplementedError


    packages = [x for x in dir(modules[type]) if not x.startswith('__') and not x.islower()]
    package_names  = [x for x in dir(modules[type]) if not x.startswith('__') and x.islower()]

    dataset = 'youcook2'
    original_text = np.concatenate([x['text'] for x in original.values()])

    if type == 'pos':
        treebanks = ['NN', 'VB', 'JJ', 'RB']
        packages = [packages[0], packages[0], packages[0], packages[0], packages[1]]
        package_names = [package_names[0], package_names[0], package_names[0], package_names[0], package_names[1]]

    pbar1 = tqdm(enumerate(zip(packages, package_names)), total=len(packages), position=0, leave=True)
    for idx, (func, perturbation) in pbar1:
        pbar = tqdm(original.items(), desc=f'{dataset} {perturbation}', total=len(original))
        if type == 'pos' and idx < len(treebanks):
            func = getattr(modules[type], func)(treebanks[idx])
            perturbation = perturbation + '_' + treebanks[idx]
        else:
            func = getattr(modules[type], func)()
        collect_data = dict()
        for video_id, annot in pbar:
            collect_text = list()
            collect_data[video_id] = {k:v for k, v in annot.items()}
            for text in annot['text']:
                if type == 'pos':
                    sentence = nltk.sent_tokenize(text)
                    for sent in sentence:
                        tags = nltk.pos_tag(nltk.word_tokenize(sent))
                    x = [x[0] for x in tags]
                    y = [y[1] for y in tags]
                    data = {'x': x, 'y': y}
                elif type == 'sentiment':
                    data = {'x': text, 'y': 'pos'}
                else:
                    data = {'x': text}

                sample = samples[type](data)
                new_text = run_transform_v2(func, sample)

                if isinstance(new_text, list):
                    new_text = ' '.join(new_text)
                collect_text.append(new_text)

            collect_data[video_id]['text'] = collect_text

        new_texts = np.concatenate([x['text'] for x in collect_data.values()])

        changed = np.sum(original_text != new_texts)
        print(f"{changed}/{len(new_texts)} changed")
        if changed > .3 * len(new_texts):
            print("Saving")
            with open(os.path.join(save_dir, f'youcookii_{perturbation}.pickle'), 'wb') as f:
                pickle.dump(collect_data, f)


def run_transform_v2(transform, text):
    try:
        trans_sample = transform.transform(text)
    except:
        if transform.__module__ == 'textflint.generation.transformation.UT.reverse_neg' and text.dump()['x'].endswith('is'):
            return text.dump()['x'] + ' not'
        else:
            pdb.set_trace()
            return text.dump()['x']
    try:
        result = trans_sample[0].dump()['x']
        return result
    except IndexError:
        print(f"Failed to get result for {text.dump()['x']} with transform {transform}")
        return text.dump()['x']


def generate_gtp3_perturbation(dataset):
    """
    Generates a generated text with the original text as seeds using a faux GPT3.
    :param dataset:
    :return:
    """
    severities = [.1, .3, .6, .9]
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B", device=0, use_fast=True)

    if dataset == 'msrvtt':
        save_dir = 'datasets/msrvtt/'
        original = 'datasets/msvrtt_for_text_eval.csv'
    elif dataset == 'youcook2':
        # For other models
        save_dir = 'datasets/youcook2/'
        original = 'datasets/youcook2_eval.csv'
    else:
        print("Unrecognized dataset.")
        return

    df = pd.read_csv(original)
    original_text = df['text'].values
    df['original_text'] = original_text
    collect = list()

    bar2 = tqdm(df.iterrows(), total=len(df))
    for idx, row in bar2:
        text = row['text']
        new_text = generator(text, max_length=90, do_sample=True, temperature=0.9)[0]['generated_text']
        collect.append(new_text)

    # For most models, save as csv
    df['text'] = collect
    df['sentence'] = collect

    df.to_csv(os.path.join(save_dir, f'{dataset}_gpt3.csv'))


def generate_gpt3_perturb_pickle():
    """
    Generates a generated text with the original text as seeds using a faux GPT3.
    This is for UniVL and only for youcook2.
    :return:
    """
    save_dir = 'data/youcook2/youcookii/'
    original = pickle.load(open('data/youcook2/youcookii/youcookii_data.no_transcript.pickle', 'rb'))
    collect = dict()
    severities = [10, 20, 30, 40, 50]
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
    for video_id, annot in tqdm(original.items(), total=len(original)):
        for sev, max_len in enumerate(severities):
            collect[sev] = dict()
            collect[sev][video_id] = annot
            tmp = list()
            for text in annot['text']:
                new_text = generator(text, max_length=max_len, do_sample=True, temperature=0.9)[0]['generated_text']
                tmp.append(new_text)
            collect[sev][video_id]['text'] = tmp

    for sev, annot in collect.items():
        with open(os.path.join(save_dir, f'youcookii_gpt3_{sev}.pickle'), 'wb') as f:
            pickle.dump(annot, f)


def generate_gender_bender_text_for_msrvtt(df, save_dir):
    from gender_bender import gender_bend
    other_mappings = get_gender_mappings()

    # Easy
    collect_swap = list()
    collect_male = list()
    collect_female = list()
    collect_neutral = list()

    count = 0
    count_male = 0
    count_female = 0
    pbar = tqdm(df.iterrows(), total=len(df), desc='Gender')
    for idx, row in pbar:
        text = row['original_text']
        new_text_swap = gender_bend(text)
        new_text_male = convert_text(text, 'male', other_mappings)
        new_text_female = convert_text(text, 'female', other_mappings)
        new_text_neutral = convert_text(text, '', other_mappings, True)

        if new_text_swap != text:
            count += 1
        if new_text_male != text:
            count_male += 1
        if new_text_female != text:
            count_female += 1
        pbar.set_postfix({'changed swap': f"{count}/{len(df)}",
                          'changed male': f"{count_male}/{len(df)}",
                          'changed female': f"{count_female}/{len(df)}"})

        collect_swap.append(new_text_swap)
        collect_male.append(new_text_male)
        collect_female.append(new_text_female)
        collect_neutral.append(new_text_neutral)

    df['text'] = collect_swap
    df.to_csv(os.path.join(save_dir, 'msrvtt_gender_swap.csv'), index=False)

    df['text'] = collect_male
    df.to_csv(os.path.join(save_dir, 'msrvtt_gender_male.csv'), index=False)

    df['text'] = collect_female
    df.to_csv(os.path.join(save_dir, 'msrvtt_gender_female.csv'), index=False)

    df['text'] = collect_neutral
    df.to_csv(os.path.join(save_dir, 'msrvtt_gender_neutral.csv'), index=False)


def convert_text(text, gender, mappings, neutral=False):
    words = text.split()
    new_words = list()
    for word in words:
        if word.lower() in mappings.keys():
            if neutral:
                new_words.append(mappings[word.lower()]['remove'])
            else:
                if mappings[word.lower()]['type'] != gender:
                    new_words.append(mappings[word.lower()]['flip'])
                else:
                    new_words.append(word)
        else:
            new_words.append(word)
    return ' '.join(new_words)


def generate_style_perturb_for_pickle(original,
                                      save_dir,
                                      perturbs=['active_to_passive', 'passive_to_active',
                                               'formal_to_casual', 'casual_to_formal'],
                                     cpu=False):

    if 'formal_to_casual' in perturbs:
        form_to_cas = Styleformer(style=1)
        form_to_cas_dict = dict()
    if 'casual_to_formal' in perturbs:
        cas_to_form = Styleformer(style=0)
        cas_to_form_dict = dict()
    if 'passive_to_active' in perturbs:
        pass_to_act = Styleformer(style=3)
        pass_to_act_dict = dict()
    if 'active_to_passive' in perturbs:
        act_to_pass = Styleformer(style=2)
        act_to_pass_dict = dict()

    for video_id, annot in tqdm(original.items(), total=len(original)):
        if 'casual_to_formal' in perturbs:
            cas_to_form_dict[video_id] = {k:v for k, v in annot.items()}
            tmp_cas_to_form = list()
        if 'formal_to_casual' in perturbs:
            form_to_cas_dict[video_id] = {k:v for k, v in annot.items()}
            tmp_form_to_cas = list()
        if 'passive_to_active' in perturbs:
            pass_to_act_dict[video_id] = {k:v for k, v in annot.items()}
            tmp_pass_to_act = list()
        if 'active_to_passive' in perturbs:
            act_to_pass_dict[video_id] = {k:v for k, v in annot.items()}
            tmp_act_to_pass = list()

        for text in annot['text']:
            if 'casual_to_formal' in perturbs:
                _cf = cas_to_form.transfer(text, inference_on=0 if cpu else 1)
                tmp_cas_to_form.append(_cf if _cf is not None else text)
            if 'formal_to_casual' in perturbs:
                _fc = form_to_cas.transfer(text, inference_on=0 if cpu else 1)
                tmp_form_to_cas.append(_fc if _fc is not None else text)
            if 'passive_to_active' in perturbs:
                _pa = pass_to_act.transfer(text, inference_on=0 if cpu else 1)
                tmp_pass_to_act.append(_pa if not None else text)
            if 'active_to_passive' in perturbs:
                _ap = act_to_pass.transfer(text, inference_on=0 if cpu else 1)
                tmp_act_to_pass.append(_ap if not None else text)

        if 'casual_to_formal' in perturbs:
            cas_to_form_dict[video_id]['text'] = tmp_cas_to_form
        if 'formal_to_casual' in perturbs:
            form_to_cas_dict[video_id]['text'] = tmp_form_to_cas
        if 'passive_to_active' in perturbs:
            pass_to_act_dict[video_id]['text'] = tmp_pass_to_act
        if 'active_to_passive' in perturbs:
            act_to_pass_dict[video_id]['text'] = tmp_act_to_pass

    if 'casual_to_formal' in perturbs:
        with open(os.path.join(save_dir, 'youcookii_casual_to_formal.pickle'), 'wb') as f:
            pickle.dump(cas_to_form_dict, f)
    if 'formal_to_casual' in perturbs:
        with open(os.path.join(save_dir, 'youcookii_formal_to_casual.pickle'), 'wb') as f:
            pickle.dump(form_to_cas_dict, f)
    if 'passive_to_active' in perturbs:
        with open(os.path.join(save_dir, 'youcookii_passive_to_active.pickle'), 'wb') as f:
            pickle.dump(pass_to_act_dict, f)
    if 'active_to_passive' in perturbs:
        with open(os.path.join(save_dir, 'youcookii_active_to_passive.pickle'), 'wb') as f:
            pickle.dump(act_to_pass_dict, f)

    logger.info("Done")


def generate_style_perturb(df,
                           save_dir,
                           dataset,
                           perturbs=['active_to_passive', 'passive_to_active', 'formal_to_casual', 'casual_to_formal'],
                           debug=False,
                           cpu=False):
    """
    Uses the styleformer to generate different textual perturbations
        * 0=Casual to Formal
        * 1=Formal to Casual
        * 2=Active to Passive
        * 3=Passive to Active

        E

    :param csv_pth:
    :return:
    """
    if debug:
        print("RUNNING IN DEBUG MODE")
        # Subsample minumum required for MUAVE
        df = df[:78]

    # Easy
    if 'formal_to_casual' in perturbs:
        form_to_cas = Styleformer(style=1)
    if 'casual_to_formal' in perturbs:
        cas_to_form = Styleformer(style=0)
    if 'passive_to_active' in perturbs:
        pass_to_act = Styleformer(style=3)
    if 'active_to_passive' in perturbs:
        act_to_pass = Styleformer(style=2)

    collect_fc = list()
    collect_cf = list()
    collect_pa = list()
    collect_ap = list()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='TextStyle'):
        text = row['text']
        if 'casual_to_formal' in perturbs:
            _cf = cas_to_form.transfer(text, inference_on=0 if cpu else 1)
            collect_cf.append(_cf if _cf is not None else text)
        if 'formal_to_casual' in perturbs:
            _fc = form_to_cas.transfer(text, inference_on=0 if cpu else 1)
            collect_fc.append(_fc if _fc is not None else text)
        if 'passive_to_active' in perturbs:
            _pa = pass_to_act.transfer(text, inference_on=0 if cpu else 1)
            collect_pa.append(_pa if not None else text)
        if 'active_to_passive' in perturbs:
            _ap = act_to_pass.transfer(text, inference_on=0 if cpu else 1)
            collect_ap.append(_ap if not None else text)

    if 'casual_to_formal' in perturbs:
        df['text'] = collect_cf
        df.to_csv(os.path.join(save_dir, f'{dataset}_casual_to_formal.csv'), index=False)
    if 'formal_to_casual' in perturbs:
        df['text'] = collect_fc
        df.to_csv(os.path.join(save_dir, f'{dataset}_formal_to_casual.csv'), index=False)
    if 'active_to_passive' in perturbs:
        df['text'] = collect_ap
        df.to_csv(os.path.join(save_dir, f'{dataset}_active_to_passive.csv'), index=False)
    if 'passive_to_active' in perturbs:
        df['text'] = collect_pa
        df.to_csv(os.path.join(save_dir, f'{dataset}_passive_to_active.csv'), index=False)
    print("Done.")


def generate_positional_drops(df, save_dir, dataset):
    """
    These will generate four perturbations evalutating the importance of positional information.
    They will replace words from the start and end with [UNK] and shuffle words so order is longer mantained.
    :param df:
    :param save_dir:
    :param dataset:
    :return:
    """
    drop_first = list()
    drop_last = list()
    shuffle_order = list()
    drop_first_and_last = list()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Positional Drops'):
        text = row[f'original_text']

        words = text.split()

        new_row_drop_first = ' '.join(['[UNK]']+words[1:])
        new_row_drop_last = ' '.join(words[:-1]+['[UNK]'])
        new_drop_first_and_last = ' '.join(['[UNK]']+words[1:-1] + ['[UNK]'])
        if len(new_drop_first_and_last) < 1:
            new_drop_first_and_last = '[UNK] [UNK]'
        random.shuffle(words)
        new_row_shuffle = ' '.join(words)

        drop_first.append(new_row_drop_first)
        drop_last.append(new_row_drop_last)
        drop_first_and_last.append(new_drop_first_and_last)
        shuffle_order.append(new_row_shuffle)

    df['text'] = drop_first
    df.to_csv(os.path.join(save_dir, f'{dataset}_drop_first.csv'), index=False)

    df['text'] = drop_last
    df.to_csv(os.path.join(save_dir, f'{dataset}_drop_last.csv'), index=False)

    df['text'] = drop_first_and_last
    df.to_csv(os.path.join(save_dir, f'{dataset}_drop_first_and_last.csv'), index=False)

    df['text'] = shuffle_order
    df.to_csv(os.path.join(save_dir, f'{dataset}_shuffle_order.csv'), index=False)


def generate_positional_drops_pickle(original, save_dir):
    collect_dict = dict()
    collect_dict['drop_first'] = dict()
    collect_dict['drop_last'] = dict()
    collect_dict['shuffle_order'] = dict()
    collect_dict['drop_first_and_last'] = dict()

    for video_id, annot in tqdm(original.items(), total=len(original), desc='Positional Drops'):
        collect_dict['drop_first'][video_id] = {k:v for k,v in annot.items()}
        collect_dict['drop_last'][video_id] = {k:v for k,v in annot.items()}
        collect_dict['shuffle_order'][video_id] = {k:v for k,v in annot.items()}
        collect_dict['drop_first_and_last'][video_id] = {k:v for k,v in annot.items()}

        collect_drop_first = list()
        collect_drop_last = list()
        collect_shuffle_order = list()
        collect_drop_first_and_last = list()

        for text in annot['text']:
            words = text.split()
            new_row_drop_first = ' '.join(['[UNK]']+words[1:])
            new_row_drop_last = ' '.join(words[:-1]+['[UNK]'])
            new_drop_first_and_last = ' '.join(['[UNK]']+words[1:-1] + ['[UNK]'])
            random.shuffle(words)
            new_row_shuffle = ' '.join(words)

            collect_drop_first.append(new_row_drop_first)
            collect_drop_last.append(new_row_drop_last)
            collect_shuffle_order.append(new_drop_first_and_last)
            collect_drop_first_and_last.append(new_row_shuffle)

        collect_dict['drop_first'][video_id]['text'] = collect_drop_first
        collect_dict['drop_last'][video_id]['text'] = collect_drop_last
        collect_dict['shuffle_order'][video_id]['text'] = collect_shuffle_order
        collect_dict['drop_first_and_last'][video_id]['text'] = collect_drop_first_and_last

    for perturb, annot in collect_dict.items():
        with open(os.path.join(save_dir, f'youcookii_{perturb}.pickle'), 'wb') as f:
            pickle.dump(annot, f)

# RETIRED AND NOT USED
def even_simpler_drops(csv_pth='datasets/youcook2_eval.csv'):
    df = pd.read_csv(csv_pth)
    df['original_text'] = df['text']
    new_text = dict()
    new_text['first_drop'] = list()
    new_text['second_drop'] = list()
    new_text['third_drop'] = list()

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row['original_text'].split()
        if len(text) > 2:
            new_text['first_drop'].append(' '.join(text[1:]))
            new_text['second_drop'].append(' '.join(text[:-1]))
            new_text['third_drop'].append(' '.join(text[1:][:-1]))
        else:
            new_text['first_drop'].append(text)
            new_text['second_drop'].append(text)
            new_text['third_drop'].append(text)

    df['text'] = new_text['first_drop']
    df.to_csv(f'datasets/youcook2_drop_first.csv', index=False)

    df['text'] = new_text['second_drop']
    df.to_csv(f'datasets/youcook2_drop_last.csv', index=False)

    df['text'] = new_text['third_drop']
    df.to_csv(f'datasets/youcook2_drop_firstlast.csv', index=False)


def simple_drops(csv_pth='datasets/youcook2_eval.csv'):
    df = pd.read_csv(csv_pth)
    df['original_text'] = df['text']
    new_text = dict()
    new_text['tagged'] = dict()
    new_text['other'] = dict()

    from nltk import pos_tag
    for sev in tqdm(range(1, 6), total=5, position=0, leave=True):
        new_text['tagged'][sev] = list()
        new_text['other'][sev] = list()
        drop_rate = [0.1, 0.2, 0.3, 0.4, 0.5][sev - 1]
        for idx, row in tqdm(df.iterrows(), total=len(df), position=1, leave=False):
            text = row['original_text']
            tags = pos_tag(text.split())

            tagged = drop_words_by_tag(tags, drop_rate)
            if tagged != -1:
                new_text['tagged'][sev].append(tagged)

            other = drop_words_despite_tag(tags, drop_rate)
            if other != -1:
                new_text['other'][sev].append(other)

        # ADD TO DATAFRAME
        df['text'] = new_text['tagged'][sev]
        df.to_csv(f'datasets/youcook2_drop_tagged_{sev}.csv', index=False)

        df['text'] = new_text['other'][sev]
        df.to_csv(f'datasets/youcook2_drop_other_{sev}.csv', index=False)
    print("Done")


def drop_words_despite_tag(tags, drop_rate):
    new_text = list()
    for x, pos in tags:
        if pos != 'VB' and pos != 'NN': # not pos.startswith('NN'):
            if np.random.uniform() < drop_rate:
                continue
            else:
                new_text.append(x)
        else:
            new_text.append(x)
    if len(new_text) < 1:

        if len(tags) < 4:
            return ' '.join([x[0] for x in tags])
        else:
            pdb.set_trace()

    else:
        return ' '.join(new_text)


def drop_words_by_tag(tags, drop_rate):
    new_text = list()
    for x, pos in tags:
        if pos.startswith('NN') or pos == 'VB':
            if np.random.uniform() < drop_rate:
                continue
            else:
                new_text.append(x)
        else:
            new_text.append(x)
    if len(new_text) < 1:
        if len(tags) < 4:
            return ' '.join([x[0] for x in tags])
        else:
            pdb.set_trace()
    else:
        return ' '.join(new_text)


def run_transform(transform, text):
    trans_sample = transform.transform(text)
    try:
        result = trans_sample[0].dump()['x']
        return result
    except IndexError:
        print(f"Failed to get result for {text.dump()['x']} with transform {transform}")
        return text.dump()['x']


def basic_text_transforms():
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running text perturbations')
    parser.add_argument('dataset', type=str)
    parser.add_argument('--pickle', const=True, default=False, action="store_const",
                        help="This is to generate files in the format necessary for the UniVL model.")
    parser.add_argument('--meta_pth', default=None, type=str)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--text_style', const=True, default=False, action="store_const")
    parser.add_argument('--textflint', const=True, default=False, action="store_const")
    parser.add_argument('--drop_pos', const=True, default=False, action="store_const")
    parser.add_argument('--posit_embd', const=True, default=False, action="store_const")
    parser.add_argument('--gender', const=True, default=False, action="store_const")

    args = parser.parse_args()

    # Create a Logger Object - Which listens to everything
    logger = logging.getLogger(os.path.basename(__file__))
    logger.setLevel(logging.DEBUG)

    # Register the Console as a handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Log format includes date and time
    formatter = logging.Formatter('%(asctime)s %(levelname)-5s %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    assert args.dataset in ['msrvtt', 'youcook2'], "Incorrect dataset passed. Must pass either `youcook2` or `msrvtt`"
    if args.dataset == 'msrvtt' and args.pickle:
        logger.warning("No pickle format for MSRVTT, ignoring pickle flag.")

    # Set up save dir
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        if args.dataset == 'msrvtt':
            save_dir = 'datasets/msrvtt_v2/'
        elif args.dataset == 'youcook2' and not args.pickle:
            save_dir = 'datasets/youcook2_v2'
        elif args.dataset == 'youcook2' and args.pickle:
            save_dir = 'data/youcook2/youcookii/'

    if not os.path.exists(save_dir):
        Path(save_dir).mkdir(parents=True)

    # Load original meta file
    if args.meta_pth is not None:
        meta_pth = args.meta_pth
        if args.pickle:
            df = pickle.load(open(meta_pth, 'rb'))
        else:
            df = pd.read_csv(meta_pth)
    else:
        if args.dataset == 'msrvtt':
            df = pd.read_csv('datasets/MSRVTT/msvrtt_eval.csv')
            df['original_text'] = df['text'].values
        elif args.dataset == 'youcook2' and not args.pickle:
            df = pd.read_csv('datasets/youcook2/youcook2_eval.csv')
            df['original_text'] = df['text'].values
        elif args.dataset == 'youcook2' and args.pickle:
            df = pickle.load(open('data/youcook2/youcookii/youcookii_data.no_transcript.pickle', 'rb'))

    if not os.path.exists(save_dir):
        Path(save_dir).mkdir(parents=True)

    if args.text_style:
        logger.info(f"Running TextStyle perturbations for {args.dataset}")
        if args.dataset == 'youcook2' and args.pickle:
            generate_style_perturb_for_pickle(df, save_dir)
        else:
            generate_style_perturb(df, save_dir, dataset=args.dataset)

    if args.textflint:
        logger.info(f"Running TextFlint perturbations for {args.dataset}")
        for type in ['universall', 'pos', 'sentiment']:
            if args.dataset == 'youcook2' and args.pickle:
                generate_textflint_pickle(df, save_dir, type)
            else:
                generate_textflint(df, save_dir, args.dataset, type)

    if args.drop_pos:
        logger.info(f"Running swap POS for UNK for {args.dataset}")
        if args.dataset == 'youcook2' and args.pickle:
            modify_pos_pickle(df, save_dir)
        else:
            modify_pos(df, save_dir, args.dataset)

    if args.gender and args.dataset == 'youcook2':
        logger.error(f"Gender perturbations are not applicable to YouCook2 dataset.")

    if args.gender and args.dataset == 'msrvtt':
        logger.info(f"Running gender perturbations for {args.dataset}")
        generate_gender_bender_text_for_msrvtt(df, save_dir)

    if args.posit_embd:
        logger.info(f"Running positional embedding-related perturbations for {args.dataset}")
        if args.dataset == 'youcook2' and args.pickle:
            generate_positional_drops_pickle(df, save_dir)
        else:
            generate_positional_drops(df, save_dir, args.dataset)
