NAME indsched_small_1
ROWS
 N  OBJ
 L  b0_1    
 L  a0_1    
 L  b0_2    
 L  a0_2    
 L  b0_3    
 L  a0_3    
 L  b0_4    
 L  a0_4    
 L  b0_5    
 L  a0_5    
 L  b1_2    
 L  a1_2    
 L  b1_3    
 L  a1_3    
 L  b1_4    
 L  a1_4    
 L  b1_5    
 L  a1_5    
 L  b2_3    
 L  a2_3    
 L  b2_4    
 L  a2_4    
 L  b2_5    
 L  a2_5    
 L  b3_4    
 L  a3_4    
 L  b3_5    
 L  a3_5    
 L  b4_5    
 L  a4_5    
COLUMNS
    s0        OBJ       9
    s0        b0_1      1
    s0        a0_1      -1
    s0        b0_2      1
    s0        a0_2      -1
    s0        b0_3      1
    s0        a0_3      -1
    s0        b0_4      1
    s0        a0_4      -1
    s0        b0_5      1
    s0        a0_5      -1
    s1        OBJ       5
    s1        b0_1      -1
    s1        a0_1      1
    s1        b1_2      1
    s1        a1_2      -1
    s1        b1_3      1
    s1        a1_3      -1
    s1        b1_4      1
    s1        a1_4      -1
    s1        b1_5      1
    s1        a1_5      -1
    s2        OBJ       5
    s2        b0_2      -1
    s2        a0_2      1
    s2        b1_2      -1
    s2        a1_2      1
    s2        b2_3      1
    s2        a2_3      -1
    s2        b2_4      1
    s2        a2_4      -1
    s2        b2_5      1
    s2        a2_5      -1
    s3        OBJ       3
    s3        b0_3      -1
    s3        a0_3      1
    s3        b1_3      -1
    s3        a1_3      1
    s3        b2_3      -1
    s3        a2_3      1
    s3        b3_4      1
    s3        a3_4      -1
    s3        b3_5      1
    s3        a3_5      -1
    s4        OBJ       9
    s4        b0_4      -1
    s4        a0_4      1
    s4        b1_4      -1
    s4        a1_4      1
    s4        b2_4      -1
    s4        a2_4      1
    s4        b3_4      -1
    s4        a3_4      1
    s4        b4_5      1
    s4        a4_5      -1
    s5        OBJ       7
    s5        b0_5      -1
    s5        a0_5      1
    s5        b1_5      -1
    s5        a1_5      1
    s5        b2_5      -1
    s5        a2_5      1
    s5        b3_5      -1
    s5        a3_5      1
    s5        b4_5      -1
    s5        a4_5      1
    MARKER    'MARKER'                 'INTORG'
    y0_1      OBJ       0
    y0_2      OBJ       0
    y0_3      OBJ       0
    y0_4      OBJ       0
    y0_5      OBJ       0
    y1_2      OBJ       0
    y1_3      OBJ       0
    y1_4      OBJ       0
    y1_5      OBJ       0
    y2_3      OBJ       0
    y2_4      OBJ       0
    y2_5      OBJ       0
    y3_4      OBJ       0
    y3_5      OBJ       0
    y4_5      OBJ       0
    MARKER    'MARKER'                 'INTEND'
RHS
    RHS1      b0_1      -18
    RHS1      a0_1      -11
    RHS1      b0_2      -18
    RHS1      a0_2      -5
    RHS1      b0_3      -18
    RHS1      a0_3      -20
    RHS1      b0_4      -18
    RHS1      a0_4      -12
    RHS1      b0_5      -18
    RHS1      a0_5      -10
    RHS1      b1_2      -11
    RHS1      a1_2      -5
    RHS1      b1_3      -11
    RHS1      a1_3      -20
    RHS1      b1_4      -11
    RHS1      a1_4      -12
    RHS1      b1_5      -11
    RHS1      a1_5      -10
    RHS1      b2_3      -5
    RHS1      a2_3      -20
    RHS1      b2_4      -5
    RHS1      a2_4      -12
    RHS1      b2_5      -5
    RHS1      a2_5      -10
    RHS1      b3_4      -20
    RHS1      a3_4      -12
    RHS1      b3_5      -20
    RHS1      a3_5      -10
    RHS1      b4_5      -12
    RHS1      a4_5      -10
BOUNDS
 LO BND1      s0        26
 LO BND1      s1        30
 LO BND1      s2        18
 LO BND1      s3        38
 LO BND1      s4        23
 LO BND1      s5        11
 BV BND1      y0_1    
 BV BND1      y0_2    
 BV BND1      y0_3    
 BV BND1      y0_4    
 BV BND1      y0_5    
 BV BND1      y1_2    
 BV BND1      y1_3    
 BV BND1      y1_4    
 BV BND1      y1_5    
 BV BND1      y2_3    
 BV BND1      y2_4    
 BV BND1      y2_5    
 BV BND1      y3_4    
 BV BND1      y3_5    
 BV BND1      y4_5    
INDICATORS
 IF  b0_1      y0_1      1
 IF  a0_1      y0_1      0
 IF  b0_2      y0_2      1
 IF  a0_2      y0_2      0
 IF  b0_3      y0_3      1
 IF  a0_3      y0_3      0
 IF  b0_4      y0_4      1
 IF  a0_4      y0_4      0
 IF  b0_5      y0_5      1
 IF  a0_5      y0_5      0
 IF  b1_2      y1_2      1
 IF  a1_2      y1_2      0
 IF  b1_3      y1_3      1
 IF  a1_3      y1_3      0
 IF  b1_4      y1_4      1
 IF  a1_4      y1_4      0
 IF  b1_5      y1_5      1
 IF  a1_5      y1_5      0
 IF  b2_3      y2_3      1
 IF  a2_3      y2_3      0
 IF  b2_4      y2_4      1
 IF  a2_4      y2_4      0
 IF  b2_5      y2_5      1
 IF  a2_5      y2_5      0
 IF  b3_4      y3_4      1
 IF  a3_4      y3_4      0
 IF  b3_5      y3_5      1
 IF  a3_5      y3_5      0
 IF  b4_5      y4_5      1
 IF  a4_5      y4_5      0
ENDATA
