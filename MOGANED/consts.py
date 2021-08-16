NONE = 'O'
PAD = "[PAD]"

train_data = "data/train.json"
dev_data = "data/dev.json"
test_data = "data/test.json"
wordemb_file = 'data/100.utf8'


batch_size = 10
lr = 0.001
n_epochs = 30

MAXLEN = 50
WORD_DIM = 100
ENTITY_DIM = 50
POSTAG_DIM = 50
POSITION_DIM = 50


# 34 event triggers
TRIGGERS = ['Business:Merge-Org',
            'Business:Start-Org',
            'Business:Declare-Bankruptcy',
            'Business:End-Org',
            'Justice:Pardon',
            'Justice:Extradite',
            'Justice:Execute',
            'Justice:Fine',
            'Justice:Trial-Hearing',
            'Justice:Sentence',
            'Justice:Appeal',
            'Justice:Convict',
            'Justice:Sue',
            'Justice:Release-Parole',
            'Justice:Arrest-Jail',
            'Justice:Charge-Indict',
            'Justice:Acquit',
            'Conflict:Demonstrate',
            'Conflict:Attack',
            'Contact:Phone-Write',
            'Contact:Meet',
            'Personnel:Start-Position',
            'Personnel:Elect',
            'Personnel:End-Position',
            'Personnel:Nominate',
            'Transaction:Transfer-Ownership',
            'Transaction:Transfer-Money',
            'Life:Marry',
            'Life:Divorce',
            'Life:Be-Born',
            'Life:Die',
            'Life:Injure',
            'Movement:Transport']

# 54 entities
ENTITIES = ['VEH:Water',
            'GPE:Nation',
            'ORG:Commercial',
            'GPE:State-or-Province',
            'Contact-Info:E-Mail',
            'Crime',
            'ORG:Non-Governmental',
            'Contact-Info:URL',
            'Sentence',
            'ORG:Religious',
            'VEH:Underspecified',
            'WEA:Projectile',
            'FAC:Building-Grounds',
            'PER:Group',
            'WEA:Exploding',
            'WEA:Biological',
            'Contact-Info:Phone-Number',
            'WEA:Chemical',
            'LOC:Land-Region-Natural',
            'WEA:Nuclear',
            'LOC:Region-General',
            'PER:Individual',
            'WEA:Sharp',
            'ORG:Sports',
            'ORG:Government',
            'ORG:Media',
            'LOC:Address',
            'WEA:Shooting',
            'LOC:Water-Body',
            'LOC:Boundary',
            'GPE:Population-Center',
            'GPE:Special',
            'LOC:Celestial',
            'FAC:Subarea-Facility',
            'PER:Indeterminate',
            'VEH:Subarea-Vehicle',
            'WEA:Blunt',
            'VEH:Land',
            'TIM:time',
            'Numeric:Money',
            'FAC:Airport',
            'GPE:GPE-Cluster',
            'ORG:Educational',
            'Job-Title',
            'GPE:County-or-District',
            'ORG:Entertainment',
            'Numeric:Percent',
            'LOC:Region-International',
            'WEA:Underspecified',
            'VEH:Air',
            'FAC:Path',
            'ORG:Medical-Science',
            'FAC:Plant',
            'GPE:Continent']

# 45 pos tags
POSTAGS = ['VBZ', 'NNS', 'JJR', 'VB', 'RBR',
           'WP', 'NNP', 'RP', 'RBS', 'VBP',
           'IN', 'UH', 'JJS', 'NNPS', 'PRP$',
           'MD', 'DT', 'WP$', 'POS', 'LS',
           'CC', 'VBN', 'EX', 'NN', 'VBG',
           'SYM', 'FW', 'TO', 'JJ', 'VBD',
           'WRB', 'CD', 'PDT', 'WDT', 'PRP',
           'RB', ',', '``', "''", ':',
           '.', '$', '#', '-LRB-', '-RRB-']

