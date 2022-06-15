#include <iostream>
#include <string>
#include <algorithm>

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventVoxel2D.h"
#include "larcv/core/DataFormat/Voxel2D.h"
#include "larcv/core/DataFormat/Voxel.h"
#include "larcv/core/DataFormat/EventParticle.h"
#include "larcv/core/PyUtil/NumpyArray.h"

int main( int nargs, char** argv )
{

  std::string dlprod_data = argv[1];
  std::string dlprod_particle = argv[2];
  //std::string out = argv[3];
  std::string out = "out.root";
  
  // take 2D Pilarnet data and then make images paired with particle set vectors
  larcv::IOManager iolcv( larcv::IOManager::kREAD );
  iolcv.add_in_file( dlprod_data );
  iolcv.add_in_file( dlprod_particle );
  iolcv.initialize();

  int nentries = iolcv.get_n_entries();
  std::cout << "number of entries: " << nentries << std::endl;

  TFile outfile( out.c_str(), "NEW" );
  TTree outtree("larclip", "LArCLIP data");
  std::vector< larcv::NumpyArrayFloat > img_v;
  std::vector< larcv::NumpyArrayFloat > caption_v;
  outtree.Branch( "image_v", &img_v );
  outtree.Branch( "caption_v", &caption_v );

  for (int ientry=0; ientry<nentries; ientry++) {

    std::cout << "////////////////////" << std::endl;
    std::cout << " Entry " << ientry << std::endl;
    std::cout << "////////////////////" << std::endl;    
    std::cout << std::endl;

    img_v.clear();
    caption_v.clear();
    
    iolcv.read_entry(ientry);
  
    // image data
    larcv::EventSparseTensor2D* ev_sparse2d
      = (larcv::EventSparseTensor2D*)iolcv.get_data( larcv::kProductSparseTensor2D, "data_zx" );

    int nimgs = ev_sparse2d->as_vector().size();
    std::cout << "number of images: " << nimgs << std::endl;

    
    larcv::EventParticle* ev_part
      = (larcv::EventParticle*)iolcv.get_data( larcv::kProductParticle, "mcst" );

    auto const& orig_meta = ev_sparse2d->as_vector().at(0).meta();
    larcv::ImageMeta meta( (float)orig_meta.cols(), (float)orig_meta.rows(),
			   orig_meta.rows(), orig_meta.cols(),
			   0, 0, 0 );
    // add missing pixel height and width
    std::cout << "meta: " << meta.dump() << std::endl;
    
    // output an image array
    larcv::NumpyArrayFloat img2d;
    img2d.ndims = 2;
    img2d.shape = std::vector<int>{ meta.cols(), meta.rows() };
    img2d.data.clear();
    img2d.data.resize( meta.cols()*meta.rows(), 0 );
    
    const larcv::SparseTensor2D& vox2d = ev_sparse2d->as_vector().at(0);
    int nvoxels = vox2d.size();    
    for ( int ivox=0; ivox<nvoxels; ivox++ ) {
      auto const& voxel = vox2d.as_vector().at(ivox);
      larcv::Point2D pt = meta.position( voxel.id() );
      int row = (int)voxel.id()/(int)meta.rows();
      int col = (int)voxel.id()%(int)meta.rows();
      //std::cout << "voxel2d[" << voxel.id() << "] x=" << pt.x << " y=" << pt.y << " r=" << row << " c=" << col << " val=" << voxel.value() << std::endl;
      img2d.data.at( voxel.id() ) = voxel.value();
    }
    img_v.emplace_back( std::move(img2d) );
    
    // take the 20 most edep particle particles
    struct Info_t {
      int idx;
      float edep;
      Info_t( int ii, float ee )
	: idx(ii),
	  edep(ee)
      {};      
      bool operator<( const Info_t& rhs ) {
	if (edep<rhs.edep) return true;
	return false;
      };
    };

    std::vector< Info_t > byenergy;
    
    for (int ipart=0; ipart<ev_part->as_vector().size(); ipart++) {
      auto const& part = ev_part->as_vector().at(ipart);
      int pid = part.pdg_code();
      float E0 = part.energy_init();
      float Edep = part.energy_deposit();
      std::cout << "part[" << ipart << "]: pid=" << pid << " Einit=" << E0 << " Edep=" << Edep << std::endl;
      if ( Edep<1.0 )
	continue;
      byenergy.push_back( Info_t(ipart,Edep) );
    }

    std::sort( byenergy.begin(), byenergy.end() );


    larcv::NumpyArrayFloat caption;
    caption.ndims = 2;
    caption.shape = std::vector<int>{ 20, 12 }; // each vector: ( edep, px, py, pz, row, col, e-, gam, mu, pro, pi, oth )
    caption.data.clear();
    caption.data.resize( 20*12, 0.0 );
    int ii=0; 
    for (auto const& info : byenergy )  {
      auto const& part = ev_part->as_vector().at( info.idx );
      
      caption.data[ ii*12 + 0 ] = part.energy_deposit();
      caption.data[ ii*12 + 1 ] = part.px()/part.p();
      caption.data[ ii*12 + 2 ] = part.py()/part.p();
      caption.data[ ii*12 + 3 ] = part.pz()/part.p();

      float z = part.position().z();
      float x = part.position().x();

      caption.data[ ii*12 + 4 ] = (float)meta.row( z )/(float)meta.rows();
      caption.data[ ii*12 + 5 ] = (float)meta.col( x )/(float)meta.cols();

      switch ( std::abs(part.pdg_code()) ) {
      case 11:
	caption.data[ ii*12 + 6 ] = 1.0;
	break;
      case 22:
	caption.data[ ii*12 + 7 ] = 1.0;
	break;
      case 13:
	caption.data[ ii*12 + 8 ] = 1.0;
	break;
      case 2212:
	caption.data[ ii*12 + 9 ] = 1.0;
	break;
      case 211:
	caption.data[ ii*12 + 10 ] = 1.0;
	break;
      default:
	caption.data[ ii*12 + 11 ] = 1.0;
	break;
      };
      
      ii++;
      if ( ii>=20 )
	break;
    }

    caption_v.emplace_back( std::move(caption) );

    outtree.Fill();
    
    if (true)
      break;
  }

  outtree.Write();
  
  return 0;
}
