proc pha2asc {phafile} {

    #Panayiotis Tzanavaris 10Oct2025
    
    #Convert a pha fits file to a two-column ascii file (counts and energy 3-30 keV)

    #This is for NuSTAR (in principle could work for any pha file with its arf, rmf). For this to work, the header of the pha file must list a background pha file, as well as an RMF and an ARF file. These three files must be in the directory where this script is run.

    file delete w1.qdp 
    data $phafile
    setplot energy
    #cpd /xs
    #plot counts

    ignore **-3.0 30.-**
    setplot command wdata w1
    plot counts
    
    


}
