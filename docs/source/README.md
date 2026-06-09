# Main goals

1. generate precise geometry diagrams from natural language 
2. Verify geometry proofs and calculations and provide corrections (increase accuracy on zhongkao/gaokao) 
3. generate valid geometry question and geometry diagram pairs
4. check other applications: CAD?

# Package organization

geoscene.add -> add geometry objects
geoscene.relate -> add relationships between objects
geoscene.constraint -> add constraints manually
geoscene.solve -> solvers
geoscene.llm -> llm operations: nl->fl, etc
geoscene.io -> save / load operations

# Tips to make it work
- GeoScene(domain=10): reduce domain
- scene.solver.numerical(penalty_weight=1): to prevent degeneracy cases

# Notes

scipy 1.13.1 faster than modern version
