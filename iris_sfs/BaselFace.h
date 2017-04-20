#pragma once
class BaselFace{
public:

/// ---- faces ---- 
static int BaselFace_faces_w;
static int BaselFace_faces_h;
static int* BaselFace_faces;

/// ---- shapeMU ---- 
static int BaselFace_shapeMU_w;
static int BaselFace_shapeMU_h;
static float* BaselFace_shapeMU;

/// ---- shapePC ---- 
static int BaselFace_shapePC_w;
static int BaselFace_shapePC_h;
static float* BaselFace_shapePC;

/// ---- shapeEV ---- 
static int BaselFace_shapeEV_w;
static int BaselFace_shapeEV_h;
static float* BaselFace_shapeEV;

/// ---- texMU ---- 
static int BaselFace_texMU_w;
static int BaselFace_texMU_h;
static float* BaselFace_texMU;

/// ---- texPC ---- 
static int BaselFace_texPC_w;
static int BaselFace_texPC_h;
static float* BaselFace_texPC;

/// ---- texEV ---- 
static int BaselFace_texEV_w;
static int BaselFace_texEV_h;
static float* BaselFace_texEV;

/// ---- segbin ---- 
static int BaselFace_segbin_w;
static int BaselFace_segbin_h;
static char* BaselFace_segbin;

/// ---- wparts ---- 
static int BaselFace_wparts_w;
static int BaselFace_wparts_h;
static float* BaselFace_wparts;

/// ---- lmInd ---- 
static int BaselFace_lmInd_w;
static int BaselFace_lmInd_h;
static int* BaselFace_lmInd;

/// ---- lmInd2 ---- 
static int BaselFace_lmInd2_w;
static int BaselFace_lmInd2_h;
static int* BaselFace_lmInd2;

/// ---- keepV ---- 
static int BaselFace_keepV_w;
static int BaselFace_keepV_h;
static char* BaselFace_keepV;

/// ---- faces_extra ---- 
static int BaselFace_faces_extra_w;
static int BaselFace_faces_extra_h;
static int* BaselFace_faces_extra;

/// ---- mid ---- 
static int BaselFace_mid_w;
static int BaselFace_mid_h;
static char* BaselFace_mid;

/// ---- texEdges ---- 
static int BaselFace_texEdges_w;
static int BaselFace_texEdges_h;
static int* BaselFace_texEdges;

/// ---- canContour ---- 
static int BaselFace_canContour_w;
static int BaselFace_canContour_h;
static char* BaselFace_canContour;

/// ---- keepVT ---- 
static int BaselFace_keepVT_w;
static int BaselFace_keepVT_h;
static int* BaselFace_keepVT;

/// ---- pair ---- 
static int BaselFace_pair_w;
static int BaselFace_pair_h;
static int* BaselFace_pair;

/// ---- pairKeepVT ---- 
static int BaselFace_pairKeepVT_w;
static int BaselFace_pairKeepVT_h;
static int* BaselFace_pairKeepVT;

/// ---- vseg_bin ---- 
static int BaselFace_vseg_bin_w;
static int BaselFace_vseg_bin_h;
static char* BaselFace_vseg_bin;

/// ---- indPX ---- 
static int BaselFace_indPX_w;
static int BaselFace_indPX_h;
static int* BaselFace_indPX;

/// ---- indNX ---- 
static int BaselFace_indNX_w;
static int BaselFace_indNX_h;
static int* BaselFace_indNX;

/// ---- symSPC ---- 
static int BaselFace_symSPC_w;
static int BaselFace_symSPC_h;
static float* BaselFace_symSPC;

/// ---- symTPC ---- 
static int BaselFace_symTPC_w;
static int BaselFace_symTPC_h;
static float* BaselFace_symTPC;

/// ---- expMU ---- 
static int BaselFace_expMU_w;
static int BaselFace_expMU_h;
static float* BaselFace_expMU;

/// ---- expEV ---- 
static int BaselFace_expEV_w;
static int BaselFace_expEV_h;
static float* BaselFace_expEV;

/// ---- expPC ---- 
static int BaselFace_expPC_w;
static int BaselFace_expPC_h;
static float* BaselFace_expPC;

/// ---- expPCFlip ---- 
static int BaselFace_expPCFlip_w;
static int BaselFace_expPCFlip_h;
static float* BaselFace_expPCFlip;
static bool load_BaselFace_data(const char* fname);
};
