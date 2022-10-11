from pathlib import Path
import pdb
import logging
import pandas as pd
import numpy as np
import random
import os
import argparse
from tqdm import tqdm

try:
    from styleformer import Styleformer
    import nltk
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


def run_gender_bender_text_for_msrvtt(df, save_dir, text_columns):
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
        new_row_swap = {k:v for k,v in row.items()}
        new_row_male = {k:v for k,v in row.items()}
        new_row_female = {k:v for k,v in row.items()}
        new_row_neutral = {k:v for k,v in row.items()}
        for col in text_columns:
            text = row[f'original_{col}']
            new_text_swap = gender_bend(text)
            new_row_swap[col] = new_text_swap

            new_text_male = convert_text(text, 'male', other_mappings)
            new_row_male[col] = new_text_male

            new_text_female = convert_text(text, 'female', other_mappings)
            new_row_female[col] = new_text_female

            new_text_neutral = convert_text(text, '', other_mappings, True)
            new_row_neutral[col] = new_text_neutral

            if new_text_swap != text:
                count += 1
            if new_text_male != text:
                count_male += 1
            if new_text_female != text:
                count_female += 1
            pbar.set_postfix({'changed swap': f"{count}/{len(df)}",
                              'changed male': f"{count_male}/{len(df)}",
                              'changed female': f"{count_female}/{len(df)}"})

        collect_swap.append(new_row_swap)
        collect_male.append(new_row_male)
        collect_female.append(new_row_female)
        collect_neutral.append(new_row_neutral)

    df = pd.DataFrame(data=collect_swap, index=range(len(collect_swap)))
    df.to_csv(os.path.join(save_dir, 'msrvtt_mc_gender_swap.csv'), index=False)

    df = pd.DataFrame(data=collect_male, index=range(len(collect_male)))
    df.to_csv(os.path.join(save_dir, 'msrvtt_mc_gender_male.csv'), index=False)

    df = pd.DataFrame(data=collect_female, index=range(len(collect_female)))
    df.to_csv(os.path.join(save_dir, 'msrvtt_mc_gender_female.csv'), index=False)

    df = pd.DataFrame(data=collect_neutral, index=range(len(collect_neutral)))
    df.to_csv(os.path.join(save_dir, 'msrvtt_mc_gender_neutral.csv'), index=False)


def run_textstyle(df, text_columns, save_dir):
    form_to_cas = Styleformer(style=1)
    act_to_pass = Styleformer(style=2)
    cas_to_form = Styleformer(style=0)

    collection = dict()
    collection['formal_to_casual'] = list()
    collection['casual_to_formal'] = list()
    collection['active_to_passive'] = list()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='TextStyle'):
        for perturb, collect in zip([cas_to_form, act_to_pass, form_to_cas], collection.keys()):
            for col in text_columns:
                text = row[f'original_{col}']
                new_text = perturb.transfer(text)
                row[col] = new_text if new_text is not None else text
            collection[collect].append(row)
    for key, collect in collection.items():
        df = pd.DataFrame(collect, index=range(len(collect)))
        df.to_csv(os.path.join(save_dir, f"msrvtt_{key}.csv"), index=False)
    logger.info("Completed 3/3 style perturbations")


def run_textflint(df, text_columns, type, save_dir):
    modules = {'universal': universal, 'pos': pos, 'machine': machine, 'sentiment': sentiment}
    samples = {'universal': UTSample, 'pos': POSSample, 'machine': MRCSample, 'sentiment': SASample}

    assert type in samples.keys(), "Passed invalid TextFlint augmentation"
    if type == 'machine':
        raise NotImplementedError

    packages = [x for x in dir(modules[type]) if not x.startswith('__') and not x.islower()]
    package_names = [x for x in dir(modules[type]) if not x.startswith('__') and x.islower()]

    if type == 'pos':
        treebanks = ['NN', 'VB', 'JJ', 'RB']
        packages = [packages[0], packages[0], packages[0], packages[0], packages[1]]
        package_names = [package_names[0], package_names[0], package_names[0], package_names[0], package_names[1]]

    pbar1 = tqdm(enumerate(zip(packages, package_names)), total=len(packages), position=0, leave=True)
    for idx, (func, perturbation) in pbar1:
        pbar = tqdm(df.iterrows(), desc=f'{perturbation}', total=len(df))
        if type == 'pos' and idx < len(treebanks):
            func = getattr(modules[type], func)(treebanks[idx])
            perturbation = perturbation + '_' + treebanks[idx]
        else:
            func = getattr(modules[type], func)()

        # Collect dataframe rows that have been altered
        collect_text = list()

        # If the file already exists, skip
        if os.path.exists(os.path.join(save_dir, f'msrvtt_mc_{perturbation}.csv')):
            logger.warning(f"File already exists {os.path.join(save_dir, f'msrvtt_mc_{perturbation}.csv')}, skipping..")
            continue

        # Iterate through dataframe rows
        for idx, row in pbar:
            for col in text_columns:
                text = row[f'original_{col}']
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
                row[col] = new_text

            # Add new row to the collection
            collect_text.append(row)

        # changed = np.sum([c['a1'] for c in collect_text] != df['original_a1'].values)
        # logger.warning(f"{changed}/{len(df['original_a1'])} changed")
        # if changed > .3 * len(df['original_a1']):
        logger.info(f"Saving {perturbation}")
        df = pd.DataFrame(data=collect_text, index=range(len(collect_text)))
        df.to_csv(os.path.join(save_dir, f'msrvtt_mc_{perturbation}.csv'), index=False)


def run_pos_changes(df, text_columns, save_dir):
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

    # Keep pos
    collect_dict['only_nouns_and_only_verbs'] = list()
    collect_dict['only_nouns'] = list()
    collect_dict['only_verbs'] = list()

    pbar = tqdm(df.iterrows(), desc=f'Change POS', total=len(df))
    for idx, row in pbar:
        new_row_no_nouns_and_no_verbs = {k:v for k,v in row.items()}
        new_row_no_nouns = {k: v for k, v in row.items()}
        new_row_no_verbs = {k: v for k, v in row.items()}

        new_row_only_nouns_and_only_verbs = {k: v for k, v in row.items()}
        new_row_only_nouns = {k: v for k, v in row.items()}
        new_row_only_verbs = {k: v for k, v in row.items()}

        for col in text_columns:
            text = row[f'original_{col}']
            no_nouns_and_no_verbs = list()
            no_nouns = list()
            no_verbs = list()
            only_nouns_and_only_verbs = list()
            only_nouns = list()
            only_verbs = list()

            sentence = nltk.sent_tokenize(text)
            for sent in sentence:
                tags = nltk.pos_tag(nltk.word_tokenize(sent))

            for word, tag in tags:
                if tag.startswith('NN'):
                    only_nouns_and_only_verbs.append(word)
                    only_nouns.append(word)
                    no_verbs.append(word)

                    no_nouns.append('[UNK]')
                    only_verbs.append('[UNK]')
                    no_nouns_and_no_verbs.append('[UNK]')
                elif tag.startswith('VB'):
                    only_nouns_and_only_verbs.append(word)
                    only_verbs.append(word)
                    no_nouns.append(word)

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

            new_row_no_nouns_and_no_verbs[col] = ' '.join(no_nouns_and_no_verbs)
            new_row_no_nouns[col] = ' '.join(no_nouns)
            new_row_no_verbs[col] = ' '.join(no_verbs)

            new_row_only_nouns_and_only_verbs[col] = ' '.join(only_nouns_and_only_verbs)
            new_row_only_nouns[col] = ' '.join(only_nouns)
            new_row_only_verbs[col] = ' '.join(only_verbs)

        # Remove pos
        collect_dict['no_nouns_and_no_verbs'].append(new_row_no_nouns_and_no_verbs)
        collect_dict['no_nouns'].append(new_row_no_nouns)
        collect_dict['no_verbs'].append(new_row_no_verbs)

        # Keep pos
        collect_dict['only_nouns_and_only_verbs'].append(new_row_only_nouns_and_only_verbs)
        collect_dict['only_nouns'].append(new_row_only_nouns)
        collect_dict['only_verbs'].append(new_row_only_verbs)

    for perturbation, new_texts in collect_dict.items():
        df = pd.DataFrame(data=new_texts, index=range(len(new_texts)))
        df.to_csv(os.path.join(save_dir, f'msrvtt_mc_{perturbation}.csv'), index=False)


def run_transform_v2(transform, text):
    try:
        trans_sample = transform.transform(text)
    except:
        if transform.__module__ == 'textflint.generation.transformation.UT.reverse_neg' and text.dump()['x'].endswith('is'):
            return text.dump()['x'] + ' not'
        else:
            return text.dump()['x']
    try:
        result = trans_sample[0].dump()['x']
        return result
    except IndexError:
        logger.error(f"Failed to get result for {text.dump()['x']} with transform {transform}")
        return text.dump()['x']


def run_positional_drops(df, text_columns, save_dir):
    drop_first = list()
    drop_last = list()
    shuffle_order = list()
    drop_first_and_last = list()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Positional Drops'):
        new_row_drop_first = {k:v for k,v in row.items()}
        new_row_drop_last = {k:v for k, v in row.items()}
        new_drop_first_and_last = {k:v for k,v in row.items()}
        new_row_shuffle = {k:v for k,v in row.items()}
        for col in text_columns:
            text = row[f'original_{col}']
            words = text.split()
            new_row_drop_first[col] = ' '.join(['[UNK]']+words[1:])
            new_row_drop_last[col] = ' '.join(words[:-1]+['[UNK]'])
            new_drop_first_and_last[col] = ' '.join(['[UNK]']+words[1:-1] + ['[UNK]'])
            random.shuffle(words)
            new_row_shuffle[col] = ' '.join(words)
        drop_first.append(new_row_drop_first)
        drop_last.append(new_row_drop_last)
        drop_first_and_last.append(new_drop_first_and_last)
        shuffle_order.append(new_row_shuffle)

    df = pd.DataFrame(data=drop_first, index=range(len(drop_first)))
    df.to_csv(os.path.join(save_dir, 'msrctt_mc_drop_first.csv'), index=False)

    df = pd.DataFrame(data=drop_last, index=range(len(drop_last)))
    df.to_csv(os.path.join(save_dir, 'msrctt_mc_drop_last.csv'), index=False)

    df = pd.DataFrame(data=drop_first_and_last, index=range(len(drop_first_and_last)))
    df.to_csv(os.path.join(save_dir, 'msrctt_mc_drop_first_and_last.csv'), index=False)

    df = pd.DataFrame(data=shuffle_order, index=range(len(shuffle_order)))
    df.to_csv(os.path.join(save_dir, 'msrctt_mc_shuffle_order.csv'), index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running text perturbations for MSRVTT-MC')
    parser.add_argument('meta_pth', default='MSRVTT-QA/MSR_MC_test.csv', type=str)
    parser.add_argument('--save_dir', default='datasets/msrvtt_mc', type=str)
    parser.add_argument('--text_style', const=True, default=False, action="store_const")
    parser.add_argument('--textflint', const=True, default=False, action="store_const")
    parser.add_argument('--drop_pos', const=True, default=False, action="store_const")
    parser.add_argument('--posit_embd', const=True, default=False, action="store_const")
    parser.add_argument('--gender', const=True, default=False, action="store_const")
    parser.add_argument('--reconstruct_mc', const=True, default=False, action="store_const")

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

    pth = args.meta_pth
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        Path(save_dir).mkdir(parents=True)

    df = pd.read_csv(pth, delimiter='\t')
    text_columns = ['a1', 'a2', 'a3', 'a4', 'a5']
    for col in text_columns:
        df[f'original_{col}'] = df[col]

    # First lets run the style perturbations
    if args.text_style:
        logger.info("Running StyleFormer Perturbations")
        run_textstyle(df, text_columns, save_dir)

    if args.textflint:
        logger.info("Running textflint perturbations")
        types = ['universal', 'pos', 'sentiment']
        for type in types:
            run_textflint(df, text_columns, type, save_dir)

    if args.drop_pos:
        logger.info("Running POS-related text perturbations")
        run_pos_changes(df, text_columns, save_dir)

    if args.posit_embd:
        logger.info("Running positional embedding-related text perturbations")
        run_positional_drops(df, text_columns, save_dir)

    if args.gender:
        logger.info("Running gender-related text perturbations")
        run_gender_bender_text_for_msrvtt(df, save_dir, text_columns)

    logger.info("Done")
