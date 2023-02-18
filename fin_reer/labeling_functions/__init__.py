from fin_reer.labeling_functions.entities.lfs import label_loc_based
from fin_reer.labeling_functions.entities.lfs import label_loc_heuristic_1
from fin_reer.labeling_functions.entities.lfs import label_loc_library_flair
from fin_reer.labeling_functions.entities.lfs import label_loc_senator
from fin_reer.labeling_functions.entities.lfs import label_org_heuristic_1
from fin_reer.labeling_functions.entities.lfs import label_org_heuristic_2
from fin_reer.labeling_functions.entities.lfs import label_org_heuristic_abbr
from fin_reer.labeling_functions.entities.lfs import label_org_heuristic_partner
from fin_reer.labeling_functions.entities.lfs import label_org_heuristic_role
from fin_reer.labeling_functions.entities.lfs import label_org_heuristic_trademark
from fin_reer.labeling_functions.entities.lfs import label_org_library_flair
from fin_reer.labeling_functions.entities.lfs import label_per_heuristic_1
from fin_reer.labeling_functions.entities.lfs import label_per_heuristic_suffix
from fin_reer.labeling_functions.entities.lfs import label_per_library_flair

entity_lfs = [label_per_library_flair,
              label_loc_library_flair,
              label_org_library_flair,
              label_per_heuristic_suffix,
              label_org_heuristic_role,
              label_org_heuristic_abbr,
              label_org_heuristic_partner,
              label_org_heuristic_trademark,
              label_loc_heuristic_1,
              label_org_heuristic_1,
              label_per_heuristic_1,
              label_org_heuristic_2,
              label_loc_senator,
              label_loc_based]
