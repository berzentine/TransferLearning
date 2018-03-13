import json
from pprint import pprint
import codecs


# extract from OSC corpus
filename = 'HA_DRCorpus/osc_hadr_corpus_v1.0_20170717_lines.json'
id = []
place = []
label = []
text= []
with open(filename, 'r') as fp:
    for lines in fp:
        d = json.loads(lines)
        if len(d['native_topic'])!=0:
            id.append(d['id'])
            temp_place = []
            temp_place.append(d['topic_country'])
            temp_place.append(d['topic_region'])
            temp_place.append(d['topic_subregion'])
            #place.append(temp_place)
            label.append(d['native_topic'].split(', ')) # should have a label and text
            text.append(d['text'].replace('\n',' '))
    fp.close()
print('Done OSC corpus')
# extract from Relief Web corpus
filename = 'HA_DRCorpus/reliefweb_corpus_raw_v1.1_20170717_lines.json'
#filename= 'relief_dummy.json'
with open(filename, 'r') as fp:
    for lines in fp:
        d = json.loads(lines)
        if len(d['disaster_type'])!=0: # should have a label and text, only then add
            id.append(d['id'])
            label.append(d['disaster_type'])
            text.append(d['text'].replace('\n',' '))
print('Done HA_DR corpus')


print('Data',len(id), len(label), len(text))
assert len(id)==len(label) and len(label)==len(text), "Unequal arrays are getting generated for data"
# write the data to the file
with codecs.open('HA_DRCorpus/en_docID.txt','w', encoding='utf-8') as idp:
    with codecs.open('HA_DRCorpus/en_label.txt','w',encoding='utf-8') as ldp:
        with codecs.open('HA_DRCorpus/en_text.txt','w',encoding='utf-8') as tdp:
            for i in range(0,len(id)):
                for l in label[i]:

                    idp.write(id[i]+"\n")
                    ldp.write(l+"\n")
                    tdp.write(text[i]+"\n")
