# Alpha-2 Country code defined by ISO 3166
COUNTRY_CODE = 'ES'

SAMPLE_TILES = [
    '31TBF', '31TCF', '31TCG', '31TDF', '31TDG'
]

RENAME = {
    'Campanya': 'Year',
    'Cultiu': 'Crop_type',
    'geometry': 'geometry',
}

CLASSES_MAPPING = {
      'AGROSTIS'                      : 'Temporary grass crops',
      'CEREALS'                       : 'Cereals',
      'CÀNEM'                         : 'Flax hemp and other similar products',
      'CASTANYERS'                    : 'Chestnuts',
      'CONREU DE FRUITS DEL BOSC'     : 'Forest',
      'ESPECIES AROMÀTIQUES HERBÀCI'  : 'Temporary medicinal etc. crops',
      'ESPECIES AROMATIQUES HERBACI'  : 'Temporary medicinal etc. crops',
      'ESPECIES AROMÀTIQUES HERBÀCIES': 'Temporary medicinal etc. crops',
      'ESPECIES AROMATIQUES LLENYOSES': 'Permanent medicinal etc. crops',
      'ESPECIES AROM�TIQUES LLENYOSE' : 'Permanent medicinal etc. crops',
      'BOIXAC O CALÈNDULA'            : 'Temporary medicinal etc. crops',
      'BOIXAC O CAL�NDULA'            : 'Temporary medicinal etc. crops',
      'BOIXAC O CALENDULA'            : 'Temporary medicinal etc. crops',
      'SAJOLIDA'                      : 'Permanent medicinal etc. crops',
      'FLORS I ORNAMENTALS'           : 'Flower crops',
      'LAVANDA O ESPÍGOL'             : 'Temporary flower crops',
      'CORIANDRE'                     : 'Temporary flower crops',
      'CAMAMILLA'                     : 'Temporary flower crops',
      'LAVANDA O ESP�GOL'             : 'Temporary flower crops',
      'LAVANDA O ESPIGOL'             : 'Temporary flower crops',
      'GARLANDA SIE'                  : 'Temporary flower crops',
      'GARLANDA'                      : 'Temporary flower crops',
      'GARLANDA NO SIE'               : 'Temporary flower crops',
      'BERBENA'                       : 'Temporary flower crops',
      'VALERIANA'                     : 'Temporary flower crops',
      'ROMANÍ'                        : 'Temporary flower crops',
      'ROMAN�'                        : 'Temporary flower crops',
      'ROMANI'                        : 'Temporary flower crops',
      'MENTA'                         : 'Other temporary spice crops n.e.c.',
      'ORENGA'                        : 'Other temporary spice crops n.e.c.',
      'FARIGOLA O TIMÓ'               : 'Other temporary spice crops n.e.c.',
      'FARIGOLA O TIM�'               : 'Other temporary spice crops n.e.c.',
      'FARIGOLA O TIMO'               : 'Other temporary spice crops n.e.c.',
      'S�LVIA'                        : 'Other temporary spice crops n.e.c.',
      'SÀLVIA'                        : 'Other temporary spice crops n.e.c.',
      'SALVIA'                        : 'Other temporary spice crops n.e.c.',
      'ALFÀBREGA'                     : 'Other temporary spice crops n.e.c.',
      'ALF�BREGA'                     : 'Other temporary spice crops n.e.c.',
      'ALFABREGA'                     : 'Other temporary spice crops n.e.c.',
      'ALFALS SIE'                    : 'Temporary grass crops',
      'ESCAIOLA'                      : 'Temporary grass crops',
      'ALFALS NO SIE'                 : 'Temporary grass crops',
      'ALFALS'                        : 'Temporary grass crops',
      'ALGARROBA HERB�CIA'            : 'Other crops permanent',
      "'ALGARROBA' HERBÀCIA NO SI"    : 'Other crops permanent',
      "'ALGARROBA' HERBÀCIA SIE"      : 'Other crops permanent',
      'QUINOA'                        : 'Other',
      'TRITORDEUM'                    : 'Other',
      'TÒFONA'                        : 'Mushrooms and truffles',
      'TOFONA'                        : 'Mushrooms and truffles',
      'BOLETS'                        : 'Mushrooms and truffles',
      'ESTEVIA'                       : 'Other sugar crops n.e.c.',
      'TABAC'                         : 'Tobacco',
      'ARRÒS'                         : 'Rice',
      'ARR�S'                         : 'Rice',
      'ARROS'                         : 'Rice',
      'BLAT DUR'                      : 'Wheat',
      'BLAT KHORASAN'                 : 'Wheat',
      'CIVADA'                        : 'Oats',
      'ESPELTA'                       : 'Wheat',
      'FAJOL'                         : 'Other',
      'ORDI'                          : 'Barley',
      'SÈGOL'                         : 'Rye',
      'S�GOL'                         : 'Rye',
      'SEGOL'                         : 'Rye',
      'TRITICALE'                     : 'Oats',
      'BLAT DE MORO'                  : 'Maize',
      'SORGO'                         : 'Sorghum',
      'CLEMENTINA'                    : 'Tangerines mandarins clementines',
      'MANDARINER HÍBRID'             : 'Tangerines mandarins clementines',
      'LLIMONER'                      : 'Lemons and Limes',
      'MANDARINER'                    : 'Tangerines mandarins clementines',
      'SATSUMA'                       : 'Tangerines mandarins clementines',
      'TARONGER'                      : 'Oranges',
      'POMELO O ARANJA'               : 'Grapefruit and pomelo',
      'CUA DE RATA (PHLEUM)'          : 'Temporary grass crops',
      'DÀCTIL (DACTYLIS)'             : 'Temporary grass crops',
      'D�CTIL (DACTYLIS)'             : 'Temporary grass crops',
      'DACTIL (DACTYLIS)'             : 'Temporary grass crops',
      'FARRATGERES'                   : 'Temporary grass crops',
      'PÈSOL I CIVADA NO SIE'         : 'Other crops temporary',
      'PÈSOL I CIVADA SIE'            : 'Other crops temporary',
      'PÈSOL I ORDI NO SIE'           : 'Other crops temporary',
      'PÈSOL I ORDI SIE'              : 'Other crops temporary',
      'RAY-GRASS'                     : 'Other crops temporary',
      'TREPADELLA NO SIE'             : 'Temporary grass crops',
      'TREPADELLA SIE'                : 'Temporary grass crops',
      'TREPADELLA'                    : 'Temporary grass crops',
      'TRÈVOL NO SIE'                 : 'Temporary grass crops',
      'TRÈVOL SIE'                    : 'Temporary grass crops',
      'TR�VOL'                        : 'Temporary grass crops',
      'TREVOL'                        : 'Temporary grass crops',
      'VEÇA I BLAT NO SIE'            : 'Other crops temporary',
      'VEÇA I BLAT SIE'               : 'Other crops temporary',
      'VEÇA I CIVADA NO SIE'          : 'Other crops temporary',
      'VEÇA I CIVADA SIE'             : 'Other crops temporary',
      'VE�A I CIVADA'                 : 'Other crops temporary',
      'VERA I CIVADA'                 : 'Other crops temporary',
      'VEÇA I ORDI NO SIE'            : 'Other crops temporary',
      'VEÇA I ORDI SIE'               : 'Other crops temporary',
      'VEÇA I TRITICALE  NO SIE'      : 'Other crops temporary',
      'VEÇA I TRITICALE SIE'          : 'Other crops temporary',
      'CANYA COMUNA'                  : 'Permanent grass crops',
      'CANYA COM�'                    : 'Permanent grass crops',
      'CANYA COMU'                    : 'Permanent grass crops',
      'ALBERCOQUERS'                  : 'Apricots',
      'KIWI'                          : 'Kiwi fruit',
      'NECTARINS'                     : 'Peaches and nectarines',
      'PERERES'                       : 'Pears and quinces',
      'PERERES/POMERES'               : 'Apples',
      'POMERES'                       : 'Apples',
      'PRESSEGUERS'                   : 'Peaches and nectarines',
      'PRESSEGUERS/NECTARINS'         : 'Peaches and nectarines',
      'PRUNERES'                      : 'Plums and sloes',
      'AMETLLERS'                     : 'Almonds',
      'AVELLANER'                     : 'Hazelnuts',
      'GARROFER'                      : 'Other fruits',
      'NOGUERES'                      : 'Walnuts',
      'PISTATXER O FESTUC'            : 'Pistachios',
      'GUARET NO SIE/ SUP. LLIURE SE*': 'Fallow land',
      'GUARET SIE AMB ESPÈCIES MEL?L' : 'Fallow land',
      'GUARET SIE/ SUP. LLIURE SEMBRA': 'Fallow land',
      'GUARET SIE AMB ESPÈCIES MEL.L*': 'Fallow land',
      'SOIA NO SIE'                   : 'Soya beans',
      'SOIA SIE'                      : 'Soya beans',
      'SOIA'                          : 'Soya beans',
      'OLIVERES'                      : 'Olives',
      'FAVES I FAVONS NO SIE'         : 'Beans',
      'FAVES I FAVONS SIE'            : 'Beans',
      'FAVES I FAVONS'                : 'Beans',
      'PÈSOLS NO SIE'                 : 'Peas',
      'PÈSOLS SIE'                    : 'Peas',
      'P�SOLS'                        : 'Peas',
      'PESOLS'                        : 'Peas',
      'TRAMUSSOS DOLÇOS SIE'          : 'Lupins',
      'TRAMUSSOS DOLÇOS NO SIE'       : 'Lupins',
      'TRAMUSSOS DOL�OS'              : 'Lupins',
      'TRAMUSSOS DOLCOS'              : 'Lupins',
      'RAÏM DE TAULA'                 : 'Grapes',
      'RA�M DE TAULA'                 : 'Grapes',
      'RAIM DE TAULA'                 : 'Grapes',
      'VINYES'                        : 'Grapes',
      'ESPINAC'                       : 'Spinach',
      'MADUIXA'                       : 'Strawberries',
      'GROSELLA'                      : 'Currants',
      'MELÓ'                          : 'Cantaloupes and other melons',
      'MEL�'                          : 'Cantaloupes and other melons',
      'PATATA'                        : 'Potatoes',
      'ENCIAM'                        : 'Lettuce',
      'ENDÍVIA, ESCAROLA I XICOIRA'   : 'Leafy or stem vegetables',
      'END�VIA, ESCAROLA I XICOIRA'   : 'Leafy or stem vegetables',
      'CRÈIXENS'                      : 'Other leafy or stem vegetables n.e.c.',
      'CR�IXENS'                      : 'Other leafy or stem vegetables n.e.c.',
      'TOMÀQUET'                      : 'Tomatoes',
      'TOM�QUET'                      : 'Tomatoes',
      'PEBROT'                        : 'Pepper (piper spp.)',
      'SÍNDRIA'                       : 'Watermelons',
      'S�NDRIA'                       : 'Watermelons',
      'PASTANAGA'                     : 'Carrots',
      'CIGRONS SIE'                   : 'Chick peas',
      'CIGRONS'                       : 'Chick peas',
      'CIGRONS NO SIE'                : 'Chick peas',
      'COL I COLIFLOR'                : 'Cauliflowers and broccoli',
      'CEBES, CALÇOTS, PORROS I ALLS' : 'Root bulb or tuberous vegetables',
      'CEBES, CAL�OTS, PORROS I ALLS' : 'Root bulb or tuberous vegetables',
      'CARXOFA'                       : 'Artichokes',
      'CARBASSÓ'                      : 'Pumpkin squash and gourds',
      'CARBASS�'                      : 'Pumpkin squash and gourds',
      'CARABASSA'                     : 'Pumpkin squash and gourds',
      'ALBERGÍNIA'                    : 'Eggplants (aubergines)',
      'ALBERG�NIA'                    : 'Eggplants (aubergines)',
      'LLENTIES SIE'                  : 'Lentils',
      'LLENTIES NO SIE'               : 'Lentils',
      'LLENTIES'                      : 'Lentils',
      'MONGETA SIE'                   : 'Beans',
      'MONGETA NO SIE'                : 'Beans',
      'MONGETA'                       : 'Beans',
      'CAMELINA'                      : 'Tea',
      'COLZA'                         : 'Rapeseed',
      'GIRA-SOL'                      : 'Sunflower',
      'MAGRANER'                      : 'Other fruits',
      'FIGUERA'                       : 'Figs',
      'FRUITERS VARIS'                : 'Other fruits',
      'CODONY'                        : 'Pears and quinces',
      'CIRERERS'                      : 'Cherries and sour cherries',
      'CAQUI'                         : 'Other tropical and subtropical fruits n.e.c.',
      'SAFR�'                         : 'Other temporary spice crops n.e.c.',
      'SAFRÀ'                         : 'Other temporary spice crops n.e.c.',
      'SAFRA'                         : 'Other temporary spice crops n.e.c.',
      'CÀRTAM O SAFRANÓ'              : 'Safflower',
      'C�RTAM O SAFRAN�'              : 'Safflower',
      'CARTAM O SAFRANS'              : 'Safflower',
      'LLÚPOL'                        : 'Other beverage crops n.e.c.',
      'LL�POL'                        : 'Other beverage crops n.e.c.',
      'LLUPOL'                        : 'Other beverage crops n.e.c.',
      'CACAUET'                       : 'Groundnuts',
      'CACAUET SIE'                   : 'Groundnuts',
      'CACAUET NO SIE'                : 'Groundnuts',
      'ESPARRECS'                     : 'Asparagus',
      'ERBS (LLEGUM) SIE'             : 'Leguminous crops n.e.c.',
      'ERBS (LLEGUM) NO SIE'          : 'Leguminous crops n.e.c.',
      'ERBS (LLEGUMINOSA)'            : 'Leguminous crops n.e.c.',
      'VECES SIE'                     : 'Leguminous crops n.e.c.',
      'VECES NO SIE'                  : 'Leguminous crops n.e.c.',
      'VECES'                         : 'Leguminous crops n.e.c.',
      'FENIGREC SIE'                  : 'Leguminous crops n.e.c.',
      'FENIGREC'                      : 'Leguminous crops n.e.c.',
      'FENIGREC NO SIE'               : 'Leguminous crops n.e.c.',
      'GUIXES NO SIE O PEDREROL NO S*': 'Leguminous crops n.e.c.',
      'GUIXES O PEDREROL'             : 'Leguminous crops n.e.c.',
      'GUIXES SIE O PEDREROL SIE'     : 'Leguminous crops n.e.c.',
      'FAVERA BORDA NO SIE O MOREU N*': 'Other crops temporary',
      'FAVERA BORDA O MOREU'          : 'Other crops temporary',
      'FAVERS BORDA SIE O MOREU SIE'  : 'Other crops temporary',
      'FESTUCA'                       : 'Permanent grass crops',
      'REMOLATXA'                     : 'Sugar beet',
      'MARDUIX'                       : 'Other crops temporary',
      'COGOMBRE'                      : 'Cucumbers',
      'JULIVERT'                      : 'Temporary flower crops',
      'NESPRER'                       : 'Other tropical and subtropical fruits n.e.c.',
      'NABIUS'                        : 'Blueberries',
      'GERDS'                         : 'Raspberries',
      'MESTALL'                       : 'Mixed cereals',
      'BORRATJA'                      : 'Temporary flower crops',
      'NASHI'                         : 'Pears and quinces',
      'FONOLL'                        : 'Other temporary spice crops n.e.c.',
      'CANEM'                         : 'Flax hemp and other similar products',
      'C�NEM'                         : 'Flax hemp and other similar products',
    # TODO                            : Remove
    # Not matched crops
    'HORTA'                         : 'Unknown crops',
    'BLAT TOU'                      : 'Unknown crops',
    'MILL'                          : 'Unknown crops',
    'BLEDA'                         : 'Unknown crops',
    'ALTRES FRUITERS'               : 'Unknown crops',
    'ALTRES PRODUCTES'              : 'Unknown crops',
    'API'                           : 'Unknown crops',
    'NAP I COL XINESA'              : 'Unknown crops',
    'RAVE'                          : 'Unknown crops',
    'VIVER ARBRE I ARBUST'          : 'Unknown crops',
    'MELISSA O TARONGINA'           : 'Unknown crops',
    'SULLA NO SIE O ENCLOVA NO SIE' : 'Unknown crops',
    'POA'                           : 'Unknown crops',
    'LLI'                           : 'Unknown crops',
    'ANET'                          : 'Unknown crops',
    'SULLA SIE O ENCLOVA SIE'       : 'Unknown crops',
    'CLOSA ANY 1'                   : 'Unknown crops',
    'PERICÓ O HERBA DE SANT JOAN'   : 'Unknown crops',
    'GUIXÓ NO SIE O PÈSOL QUADRAT'  : 'Unknown crops',
    'HISOP'                         : 'Unknown crops',
    'RETIRADA MIN. 20 ANYS (RAMSAR)': 'Unknown crops',
    'DALL O FROMENTAL (ARRHENANTHE*': 'Unknown crops',
    'DALL O FROMENTAL (ARRHENATHER*': 'Unknown crops',
    'HORTICOLES'                    : 'Unknown crops',
    'PROTEAGINOSES'                 : 'Unknown crops',
}
