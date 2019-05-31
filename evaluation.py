from __future__ import print_function

import json
import os
import pickle

import numpy
from data import get_test_loader
import time
import numpy as np
from vocab import Vocabulary  # NOQA
import torch
from model import VSE, order_sim
from collections import OrderedDict
import stanfordnlp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    all_img_ids = None
    all_captions = None
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            images, captions, lengths, ids, img_ids = batch_data

            # make sure val logger is used
            model.logger = val_logger

            # compute the embeddings
            img_emb, cap_emb = model.forward_emb(images, captions, lengths)

            # initialize the numpy arrays given the size of the embeddings
            if img_embs is None:
                img_embs = np.zeros(
                    (len(data_loader.dataset), img_emb.size(1)))
                cap_embs = np.zeros(
                    (len(data_loader.dataset), cap_emb.size(1)))
                all_captions = np.zeros(len(data_loader.dataset), dtype=object)
                all_img_ids = np.zeros(len(data_loader.dataset))

            # preserve the embeddings by copying from GPU
            # and converting to NumPy
            img_embs[ids] = img_emb.data.cpu().numpy().copy()
            cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

            #preserve also the captions and image coco ids
            all_captions[ids] = list(captions.cpu().numpy().copy())
            all_img_ids[ids] = img_ids

            # measure accuracy and record loss
            model.forward_loss(img_emb, cap_emb)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_step == 0:
                logging('Test: [{0}/{1}]\t'
                        '{e_log}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        .format(
                            i, len(data_loader), batch_time=batch_time,
                            e_log=str(model.logger)))
            del images, captions

    return img_embs, cap_embs, all_img_ids, all_captions


def evalrank(model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    with open(os.path.join(opt.vocab_path,
                           '%s_vocab.pkl' % opt.data_name), 'rb') as f:
        vocab = pickle.load(f)
    opt.vocab_size = len(vocab)

    # construct model
    model = VSE(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.crop_size,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs, cap_embs = encode_data(model, data_loader)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if not fold5:
        # no cross-validation, full evaluation
        r, rt = i2t(img_embs, cap_embs, measure=opt.measure, return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs,
                      measure=opt.measure, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000],
                         cap_embs[i * 5000:(i + 1) *
                                  5000], measure=opt.measure,
                         return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000],
                           cap_embs[i * 5000:(i + 1) *
                                    5000], measure=opt.measure,
                           return_ranks=True)
            if i == 0:
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
        print(npts)
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
        print(npts)
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

PAIR_OCCURENCES = "pair_occurrences"
OCCURRENCE_DATA = "adjective_noun_occurrence_data"
DATA_COCO_SPLIT = "coco_split"
NOUNS = "nouns"
ADJECTIVES = "adjectives"
VERBS = "verbs"

RELATION_NOMINAL_SUBJECT = "nsubj"
RELATION_ADJECTIVAL_MODIFIER = "amod"
RELATION_CONJUNCT = "conj"
RELATION_RELATIVE_CLAUSE_MODIFIER = "acl:relcl"
RELATION_ADJECTIVAL_CLAUSE = "acl"

def average_recall(recall_scores):
    pair_recalls_summed = 0

    for pair in recall_scores.keys():
        average_pair_recall = np.sum(
            list(recall_scores[pair]["true_positives"].values())
        ) / np.sum(list(recall_scores[pair]["numbers"].values()))
        pair_recalls_summed += average_pair_recall

    recall = pair_recalls_summed / len(recall_scores)
    return recall

def get_ranking_splits_from_occurrences_data(occurrences_data_files):
    evaluation_indices = []

    for file in occurrences_data_files:
        with open(file, "r") as json_file:
            occurrences_data = json.load(json_file)

        evaluation_indices.extend(
            [
                int(key)
                for key, value in occurrences_data[OCCURRENCE_DATA].items()
                if value[PAIR_OCCURENCES] >= 1 and value[DATA_COCO_SPLIT] == "val2014"
            ]
        )
    return evaluation_indices

def get_adjectives_for_noun(pos_tagged_caption, nouns):
    dependencies = pos_tagged_caption.dependencies

    adjectives = {
        d[2].lemma
        for d in dependencies
        if d[1] == RELATION_ADJECTIVAL_MODIFIER
        and d[0].lemma in nouns
        and d[2].upos == "ADJ"
    } | {
        d[0].lemma
        for d in dependencies
        if d[1] == RELATION_NOMINAL_SUBJECT
        and d[2].lemma in nouns
        and d[0].upos == "ADJ"
    }
    conjuncted_adjectives = set()
    for adjective in adjectives:
        conjuncted_adjectives.update(
            {
                d[2].lemma
                for d in dependencies
                if d[1] == RELATION_CONJUNCT
                and d[0].lemma == adjective
                and d[2].upos == "ADJ"
            }
            | {
                d[2].lemma
                for d in dependencies
                if d[1] == RELATION_ADJECTIVAL_MODIFIER
                and d[0].lemma == adjective
                and d[2].upos == "ADJ"
            }
        )
    return adjectives | conjuncted_adjectives


def get_verbs_for_noun(pos_tagged_caption, nouns):
    dependencies = pos_tagged_caption.dependencies

    verbs = (
        {
            d[0].lemma
            for d in dependencies
            if d[1] == RELATION_NOMINAL_SUBJECT
            and d[2].lemma in nouns
            and d[0].upos == "VERB"
        }
        | {
            d[2].lemma
            for d in dependencies
            if d[1] == RELATION_RELATIVE_CLAUSE_MODIFIER
            and d[0].lemma in nouns
            and d[2].upos == "VERB"
        }
        | {
            d[2].lemma
            for d in dependencies
            if d[1] == RELATION_ADJECTIVAL_CLAUSE
            and d[0].lemma in nouns
            and d[2].upos == "VERB"
        }
    )

    return verbs


def contains_adjective_noun_pair(pos_tagged_caption, nouns, adjectives):
    noun_is_present = False
    adjective_is_present = False

    for word in pos_tagged_caption.words:
        if word.lemma in nouns:
            noun_is_present = True
        if word.lemma in adjectives:
            adjective_is_present = True

    caption_adjectives = get_adjectives_for_noun(pos_tagged_caption, nouns)
    combination_is_present = bool(set(adjectives) & caption_adjectives)

    return noun_is_present, adjective_is_present, combination_is_present


def contains_verb_noun_pair(pos_tagged_caption, nouns, verbs):
    noun_is_present = False
    verb_is_present = False

    for word in pos_tagged_caption.words:
        if word.lemma in nouns:
            noun_is_present = True
        if word.lemma in verbs:
            verb_is_present = True

    caption_verbs = get_verbs_for_noun(pos_tagged_caption, nouns)
    combination_is_present = bool(set(verbs) & caption_verbs)

    return noun_is_present, verb_is_present, combination_is_present



def eval_compositional_splits(model_path, data_path, split, dataset_split):
    checkpoint = torch.load(model_path, map_location=device)
    opt = checkpoint['opt']
    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    with open(os.path.join(opt.vocab_path,
                           '%s_vocab.pkl' % opt.data_name), 'rb') as f:
        vocab = pickle.load(f)
    opt.vocab_size = len(vocab)

    # construct model
    model = VSE(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    opt.data_name = "coco"
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.crop_size,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    embedded_images, embedded_captions, all_img_ids, all_captions = encode_data(model, data_loader)
    print('Images: %d, Embedded Captions: %d, Image IDs: %d, captions: %d ' %
          (embedded_images.shape[0] / 5, embedded_captions.shape[0], all_img_ids.shape[0], all_captions.shape[0]))

    print("sample image ids: ")
    print(all_img_ids[:10])
    print("Recall@5 of pairs:")
    print(
        "Pair | Recall (n=1) | Recall (n=2) | Recall (n=3) | Recall (n=4) | Recall (n=5)"
    )
    nlp_pipeline = stanfordnlp.Pipeline()


    dataset_splits_dict = json.load(open(dataset_split, "r"))
    heldout_pairs = dataset_splits_dict["heldout_pairs"]

    recall_scores = {}

    for pair in heldout_pairs:
        print("\n\n Eval for pair: ", pair)
        occurrences_data_file = os.path.join(
            "occurrences", pair + ".json"
        )

        evaluation_indices = get_ranking_splits_from_occurrences_data([occurrences_data_file])
        occurrences_data = json.load(open(occurrences_data_file, "r"))

        nouns = set(occurrences_data[NOUNS])
        if ADJECTIVES in occurrences_data:
            adjectives = set(occurrences_data[ADJECTIVES])
        elif VERBS in occurrences_data:
            verbs = set(occurrences_data[VERBS])
        else:
            raise ValueError("No adjectives or verbs found in occurrences data!")

        true_positives = dict.fromkeys(["N=1", "N=2", "N=3", "N=4", "N=5"], 0)
        numbers = dict.fromkeys(["N=1", "N=2", "N=3", "N=4", "N=5"], 0)
        for i, coco_id in enumerate(evaluation_indices):
            print("\nCOCO IMG ID: ", coco_id)
            # Create test dataset
            indices_correct_captions = [i for i, x in enumerate(all_img_ids) if x == coco_id]
            print("Correct captions: ", indices_correct_captions)
            for j in indices_correct_captions:
                encoded_caption = list(all_captions[j])
                decoded_caption = " ".join([vocab.idx2word[ind] for ind in encoded_caption if not (
                        vocab.idx2word[ind] == "<pad>" or vocab.idx2word[ind] == "<start>" or vocab.idx2word[
                        ind] == "<end>")])
                print(decoded_caption)

            target_captions_embedded = np.concatenate(([embedded_captions[i] for i in indices_correct_captions], embedded_captions[-5000:]), axis=0)
            target_captions = np.concatenate(([all_captions[i] for i in indices_correct_captions], all_captions[-5000:]), axis=0)

            image = embedded_images[np.where(all_img_ids == coco_id)][0]

            # Compute similarity of image to all captions
            d = np.dot(image, target_captions_embedded.T).flatten()
            inds = np.argsort(d)[::-1]

            count = occurrences_data[OCCURRENCE_DATA][str(coco_id)][PAIR_OCCURENCES]

            # Look for pair occurrences in top 5 captions
            hit = False
            for j in inds[:5]:
                encoded_caption = list(target_captions[j])
                decoded_caption = " ".join([vocab.idx2word[ind] for ind in encoded_caption if not (vocab.idx2word[ind] == "<pad>" or vocab.idx2word[ind] == "<start>" or vocab.idx2word[ind] == "<end>")])
                print(decoded_caption)
                pos_tagged_caption = nlp_pipeline(decoded_caption).sentences[0]
                contains_pair = False
                if ADJECTIVES in occurrences_data:
                    _, _, contains_pair = contains_adjective_noun_pair(
                        pos_tagged_caption, nouns, adjectives
                    )
                elif VERBS in occurrences_data:
                    _, _, contains_pair = contains_verb_noun_pair(
                        pos_tagged_caption, nouns, verbs
                    )
                if contains_pair:
                    hit = True

            if hit:
                true_positives["N={}".format(count)] += 1
            numbers["N={}".format(count)] += 1

        recall_scores[pair] = {}
        recall_scores[pair]["true_positives"] = true_positives
        recall_scores[pair]["numbers"] = numbers

        average_pair_recall = np.sum(
            list(recall_scores[pair]["true_positives"].values())
        ) / np.sum(list(recall_scores[pair]["numbers"].values()))
        print("{}: {}".format(pair, np.round(average_pair_recall, 2)))

    print("Average Recall: {}".format(average_recall(recall_scores)))
    json.dump(recall_scores, open("recall_scores_"+str(os.path.basename(dataset_split)), "w"))
