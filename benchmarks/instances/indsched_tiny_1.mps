NAME indsched_tiny_1
ROWS
 N  OBJ
 L  b0_1    
 L  a0_1    
 L  b0_2    
 L  a0_2    
 L  b0_3    
 L  a0_3    
 L  b1_2    
 L  a1_2    
 L  b1_3    
 L  a1_3    
 L  b2_3    
 L  a2_3    
COLUMNS
    s0        OBJ       1
    s0        b0_1      1
    s0        a0_1      -1
    s0        b0_2      1
    s0        a0_2      -1
    s0        b0_3      1
    s0        a0_3      -1
    s1        OBJ       1
    s1        b0_1      -1
    s1        a0_1      1
    s1        b1_2      1
    s1        a1_2      -1
    s1        b1_3      1
    s1        a1_3      -1
    s2        OBJ       9
    s2        b0_2      -1
    s2        a0_2      1
    s2        b1_2      -1
    s2        a1_2      1
    s2        b2_3      1
    s2        a2_3      -1
    s3        OBJ       5
    s3        b0_3      -1
    s3        a0_3      1
    s3        b1_3      -1
    s3        a1_3      1
    s3        b2_3      -1
    s3        a2_3      1
    MARKER    'MARKER'                 'INTORG'
    y0_1      OBJ       0
    y0_2      OBJ       0
    y0_3      OBJ       0
    y1_2      OBJ       0
    y1_3      OBJ       0
    y2_3      OBJ       0
    MARKER    'MARKER'                 'INTEND'
RHS
    RHS1      b0_1      -4
    RHS1      a0_1      -17
    RHS1      b0_2      -4
    RHS1      a0_2      -8
    RHS1      b0_3      -4
    RHS1      a0_3      -12
    RHS1      b1_2      -17
    RHS1      a1_2      -8
    RHS1      b1_3      -17
    RHS1      a1_3      -12
    RHS1      b2_3      -8
    RHS1      a2_3      -12
BOUNDS
 LO BND1      s0        22
 LO BND1      s2        22
 LO BND1      s3        17
 BV BND1      y0_1    
 BV BND1      y0_2    
 BV BND1      y0_3    
 BV BND1      y1_2    
 BV BND1      y1_3    
 BV BND1      y2_3    
INDICATORS
 IF  b0_1      y0_1      1
 IF  a0_1      y0_1      0
 IF  b0_2      y0_2      1
 IF  a0_2      y0_2      0
 IF  b0_3      y0_3      1
 IF  a0_3      y0_3      0
 IF  b1_2      y1_2      1
 IF  a1_2      y1_2      0
 IF  b1_3      y1_3      1
 IF  a1_3      y1_3      0
 IF  b2_3      y2_3      1
 IF  a2_3      y2_3      0
ENDATA
