/**
 * @file  RenderView3D.cpp
 * @brief View for rendering.
 *
 */
/*
 * Original Author: Ruopeng Wang
 * CVS Revision Info:
 *    $Author: rpwang $
 *    $Date: 2010/02/03 19:33:24 $
 *    $Revision: 1.26 $
 *
 * Copyright (C) 2008-2009,
 * The General Hospital Corporation (Boston, MA).
 * All rights reserved.
 *
 * Distribution, usage and copying of this software is covered under the
 * terms found in the License Agreement file named 'COPYING' found in the
 * FreeSurfer source code root directory, and duplicated here:
 * https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferOpenSourceLicense
 *
 * General inquiries: freesurfer@nmr.mgh.harvard.edu
 * Bug reports: analysis-bugs@nmr.mgh.harvard.edu
 *
 */

#include "RenderView3D.h"
#include "MainWindow.h"
#include "ConnectivityData.h"
#include "LayerCollection.h"
#include "LayerMRI.h"
#include <vtkRenderer.h>
#include "vtkConeSource.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"
#include "vtkCamera.h"
#include "vtkActor2D.h"
#include "vtkCellPicker.h"
#include "vtkPointPicker.h"
#include "vtkPropPicker.h"
#include "vtkProp3DCollection.h"
#include "vtkScalarBarActor.h"
#include "Interactor3DNavigate.h"
#include "LayerSurface.h"
#include "SurfaceOverlayProperties.h"
#include "SurfaceOverlay.h"
#include "vtkRGBAColorTransferFunction.h"
#include "Cursor3D.h"

IMPLEMENT_DYNAMIC_CLASS(RenderView3D, RenderView)

BEGIN_EVENT_TABLE(RenderView3D, RenderView)

END_EVENT_TABLE()

RenderView3D::RenderView3D() : RenderView()
{
  InitializeRenderView3D();
}

RenderView3D::RenderView3D( wxWindow* parent, int id ) : RenderView( parent, id )
{
  InitializeRenderView3D();
}

void RenderView3D::InitializeRenderView3D()
{
  this->SetDesiredUpdateRate( 5000 );
// this->SetStillUpdateRate( 0.5 );

  if ( m_interactor )
    delete m_interactor;

  m_interactor = new Interactor3DNavigate();

  m_bToUpdateRASPosition = false;
  m_bToUpdateCursorPosition = false;
  m_bToUpdateConnectivity = false;

  vtkCellPicker* picker = vtkCellPicker::New();
// vtkPointPicker* picker = vtkPointPicker::New();
// vtkPropPicker* picker = vtkPropPicker::New();
  picker->SetTolerance( 0.0001 );
  this->SetPicker( picker );
  picker->Delete();

  for ( int i = 0; i < 3; i++ )
    m_bSliceVisibility[i] = true;

  m_cursor3D = new Cursor3D( this );
  
  m_actorScalarBar->SetNumberOfLabels( 4 );
}

RenderView3D* RenderView3D::New()
{
  // we don't make use of the objectfactory, because we're not registered
  return new RenderView3D;
}

RenderView3D::~RenderView3D()
{
  delete m_cursor3D;
}

void RenderView3D::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

void RenderView3D::RefreshAllActors()
{
  LayerCollectionManager* lcm = MainWindow::GetMainWindowPointer()->GetLayerCollectionManager();

  m_renderer->RemoveAllViewProps();
  bool b[3] = { true, true, true };
  lcm->Append3DProps( m_renderer, b );

  m_cursor3D->AppendActor( m_renderer );

  // add focus frame
  m_renderer->AddViewProp( m_actorFocusFrame );

  if ( lcm->HasLayer( "MRI" ) || lcm->HasLayer( "Surface" ) )
  {
    m_renderer->AddViewProp( m_actorScalarBar );    
  }
  
  MainWindow::GetMainWindowPointer()->GetConnectivityData()->AppendProps( m_renderer );
  
  m_renderer->ResetCameraClippingRange();

  NeedRedraw();
  // Render();
}

void RenderView3D::UpdateViewByWorldCoordinate()
{
  vtkCamera* cam = m_renderer->GetActiveCamera();
  double wcenter[3];
  for ( int i = 0; i < 3; i++ )
  {
    wcenter[i] = m_dWorldOrigin[i] + m_dWorldSize[i] / 2;
  }
  cam->SetFocalPoint( wcenter );
  cam->SetPosition( wcenter[0] - ( m_dWorldSize[1] > m_dWorldSize[2] ? m_dWorldSize[1] : m_dWorldSize[2] ) *2.5,
                    wcenter[1], 
                    wcenter[2]);
  cam->SetViewUp( 0, 0, 1 );
  m_renderer->ResetCameraClippingRange();
}

// snap the camera to the nearest axis
void RenderView3D::SnapToNearestAxis()
{
  vtkCamera* cam = m_renderer->GetActiveCamera();
  double v[3], v_up[3];
  cam->OrthogonalizeViewUp();
  cam->GetDirectionOfProjection(v);
  cam->GetViewUp(v_up);
  double wcenter[3];
  for ( int i = 0; i < 3; i++ )
  {
    wcenter[i] = m_dWorldOrigin[i] + m_dWorldSize[i] / 2;
  }
  cam->SetFocalPoint( wcenter );
  
  if ( fabs(v[0]) > fabs(v[1]) && fabs(v[0]) > fabs(v[2]) )
  {
    v[0] = ( v[0] > 0 ? 1 : -1 );
    v[1] = v[2] = 0;
  }
  else if ( fabs(v[1]) > fabs(v[2]) )
  {
    v[1] = ( v[1] > 0 ? 1 : -1 );
    v[0] = v[2] = 0;
  }
  else
  {
    v[2] = ( v[2] > 0 ? 1 : -1 );
    v[0] = v[1] = 0;
  }
  
  if ( fabs(v_up[0]) > fabs(v_up[1]) && fabs(v_up[0]) > fabs(v_up[2]) )
  {
    v_up[0] = ( v_up[0] > 0 ? 1 : -1 );
    v_up[1] = v_up[2] = 0;
  }
  else if ( fabs(v_up[1]) > fabs(v_up[2]) )
  {
    v_up[1] = ( v_up[1] > 0 ? 1 : -1 );
    v_up[0] = v_up[2] = 0;
  }
  else
  {
    v_up[2] = ( v_up[2] > 0 ? 1 : -1 );
    v_up[0] = v_up[1] = 0;
  }
  
  double pos[3];
  for ( int i = 0; i < 3; i++ )
  {
    pos[i] = wcenter[i] - ( m_dWorldSize[i] * v[i] * 2.5 );
  }
  cam->SetPosition( pos );
  cam->SetViewUp( v_up );
  m_renderer->ResetCameraClippingRange();
  
  NeedRedraw();
}


void RenderView3D::UpdateMouseRASPosition( int posX, int posY )
{
  m_bToUpdateRASPosition = true;
  m_nPickCoord[0] = posX;
  m_nPickCoord[1] = posY;
}

void RenderView3D::CancelUpdateMouseRASPosition()
{
  m_bToUpdateRASPosition = false;
}

void RenderView3D::DoUpdateRASPosition( int posX, int posY, bool bCursor )
{
  LayerCollection* lc_mri = MainWindow::GetMainWindowPointer()->GetLayerCollection( "MRI" );
  LayerCollection* lc_roi = MainWindow::GetMainWindowPointer()->GetLayerCollection( "ROI" );
  LayerCollection* lc_surface = MainWindow::GetMainWindowPointer()->GetLayerCollection( "Surface" );

// MousePositionToRAS( posX, posY, pos );
// vtkPointPicker* picker = vtkPointPicker::SafeDownCast( this->GetPicker() );
  vtkCellPicker* picker = vtkCellPicker::SafeDownCast( this->GetPicker() );
// vtkPropPicker* picker = vtkPropPicker::SafeDownCast( this->GetPicker() );
  if ( picker )
  {
    double pos[3];
    picker->Pick( posX, GetClientSize().GetHeight() - posY, 0, GetRenderer() );
    picker->GetPickPosition( pos );

    vtkProp* prop = picker->GetViewProp();
    // cout << pos[0] << " " << pos[1] << " " << pos[2] << ",   " << prop << endl;
    if ( prop && ( lc_mri->HasProp( prop ) || lc_roi->HasProp( prop ) || lc_surface->HasProp( prop ) ) )
    {
      if ( bCursor )
      {
        lc_mri->SetCursorRASPosition( pos );
        MainWindow::GetMainWindowPointer()->GetLayerCollectionManager()->SetSlicePosition( pos );
      }
      else
        lc_mri->SetCurrentRASPosition( pos );
    }
  }
}

void RenderView3D::DoUpdateConnectivityDisplay()
{
  ConnectivityData* conn = MainWindow::GetMainWindowPointer()->GetConnectivityData();
  if ( conn->IsValid() && conn->GetDisplayMode() != ConnectivityData::DM_All )
  {
    conn->BuildConnectivityActors();
//    NeedRedraw();
  }
}

void RenderView3D::UpdateCursorRASPosition( int posX, int posY )
{
  m_bToUpdateCursorPosition = true;
  m_nCursorCoord[0] = posX;
  m_nCursorCoord[1] = posY;
}


void RenderView3D::UpdateConnectivityDisplay()
{
  m_bToUpdateConnectivity = true;
}


void RenderView3D::OnInternalIdle()
{
  RenderView::OnInternalIdle();

  if ( m_bToUpdateRASPosition )
  {
    DoUpdateRASPosition( m_nPickCoord[0], m_nPickCoord[1] );
    m_bToUpdateRASPosition = false;
  }
  if ( m_bToUpdateCursorPosition )
  {
    DoUpdateRASPosition( m_nCursorCoord[0], m_nCursorCoord[1], true );
    m_bToUpdateCursorPosition = false;
  }
  if ( m_bToUpdateConnectivity )
  {
    DoUpdateConnectivityDisplay();
    m_bToUpdateConnectivity = false;
  }
}

void RenderView3D::DoListenToMessage ( std::string const iMsg, void* iData, void* sender )
{
  if ( iMsg == "CursorRASPositionChanged" )
  {
    LayerCollection* lc = MainWindow::GetMainWindowPointer()->GetLayerCollection( "MRI" );
    m_cursor3D->SetPosition( lc->GetCursorRASPosition() );
  }
  else if ( iMsg == "ConnectivityActorUpdated" )
  {
    m_renderer->ResetCameraClippingRange();
    NeedRedraw();
  }

  RenderView::DoListenToMessage( iMsg, iData, sender );
}

void RenderView3D::ShowVolumeSlice( int nPlane, bool bShow )
{
  m_bSliceVisibility[nPlane] = bShow;
  RefreshAllActors();
}

void RenderView3D::PreScreenshot()
{
  LayerCollectionManager* lcm = MainWindow::GetMainWindowPointer()->GetLayerCollectionManager();

  m_renderer->RemoveAllViewProps();
  lcm->Append3DProps( m_renderer );

  // add coordinate annotation
  SettingsScreenshot s = MainWindow::GetMainWindowPointer()->GetScreenshotSettings();
  if ( !s.HideCursor )
    m_cursor3D->AppendActor( m_renderer );

  MainWindow::GetMainWindowPointer()->GetConnectivityData()->AppendProps( m_renderer );
  
  // add scalar bar
  m_renderer->AddViewProp( m_actorScalarBar );
}

void RenderView3D::PostScreenshot()
{
  RefreshAllActors();
}

void RenderView3D::UpdateScalarBar()
{
  LayerSurface* surf = (LayerSurface*) MainWindow::GetMainWindowPointer()->GetLayerCollection( "Surface" )->GetActiveLayer();
  if ( surf && surf->GetActiveOverlay() )
  {
    m_actorScalarBar->SetLookupTable( surf->GetActiveOverlay()->GetProperties()->GetLookupTable() );
  }
  else
    RenderView::UpdateScalarBar();
}

void RenderView3D::Azimuth( double angle )
{
  vtkCamera* cam = m_renderer->GetActiveCamera();
  cam->Azimuth( angle );
  NeedRedraw();
}
