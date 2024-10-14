# msy_fisheries

***MSYfish*** is the individual based population model based on principles of dynamic energy budget theory and metabolic scaling used in Woodson et al (2024).

To run model, set input and output directories in **calc_msy_parallel_serve.py**. This file is the executable and will call other components. Inputs for an individual species can be added to the input file 'fish_growth_data2' or a new file can be created using FishLife or rfishbase.

C.B. Woodson, S.Y. Litvin, J.R. Schramski, S.B. Joye. (2024) An individual-based model for exploration of population and stock dynamics in marine fishes, Ecological Modelling 498, 110842, https://doi.org/10.1016/j.ecolmodel.2024.110842.

Abstract: Many size- or age-structure fisheries models require estimation of fundamental population level parameters such as growth, mortality, and recruitment rates that are notoriously difficult to estimate and can constrain the ability of models for exploring emergent properties in population dynamics. To address some of these issues, we develop a discrete-time individual-based model that integrates both age- and size-based concepts. Individual fish are tracked throughout their lifetime allowing for assessment of age-based concepts, with traits determined by size. This method utilizes individual growth parameters as opposed to population level growth rates and allows for many properties of populations that are normally prescribed to be emergent properties of the model. We demonstrate the utility of the model for reproducing population level parameters such as slope at origin for recruitment curves and intrinsic growth rates. The addition of spatial dynamics where a population is sub-divided into discrete stocks further allows for the assessment of various conservation techniques such as marine protected areas, fishing area rotation, and size limits at the individual level.
Keywords: Agent-based modeling; Sized-based traits; Population dynamics; Stock dynamics; Marine fishes
